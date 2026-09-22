"""Consumer checkpoints over a session's published reading stream.

A checkpoint freezes a consumer's position in the deterministic published
order ``(observed_at, source_id, seq)`` together with the published watermark
visible in the *same* read-only snapshot.  Resuming from a checkpoint pages
through whatever is published strictly after the frozen position, so the
checkpoint is an immutable bookmark: later publishes -- and closing the
session -- never change what it points at.

* Names are unique per session.  Re-creating an existing name at the same
  position (same snapshot tail) is idempotent and returns the stored
  checkpoint; the same name at a different position is a 409 conflict.
* Resume reads reuse the ordinary published-stream keyset pagination and
  ordering.  A checkpoint over an empty stream (``position = null``) resumes
  from the first published reading.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from .database import read_transaction, transaction, utc_now
from .errors import ConflictError, NotFoundError
from .service import _load_session, _published_page, _reading_out
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = [
    "create_checkpoint",
    "list_checkpoints",
    "get_checkpoint",
    "delete_checkpoint",
    "list_from_checkpoint",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _position_of_last_published(
    conn: sqlite3.Connection, session_id: str
) -> Optional[tuple[str, str, int]]:
    """Tail of the published stream in merge order inside the open snapshot."""
    row = conn.execute(
        "SELECT observed_at, source_id, seq FROM readings"
        " WHERE session_id = ? AND status = 'published'"
        " ORDER BY observed_at DESC, source_id DESC, seq DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return row["observed_at"], row["source_id"], row["seq"]


def _load_checkpoint(
    conn: sqlite3.Connection, session_id: str, name: str
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM stream_checkpoints WHERE session_id = ? AND checkpoint_name = ?",
        (session_id, name),
    ).fetchone()
    if row is None:
        raise NotFoundError(
            f"checkpoint {name!r} not found for session {session_id!r}"
        )
    return row


def _same_position(
    row: sqlite3.Row, position: Optional[tuple[str, str, int]]
) -> bool:
    if position is None:
        return row["position_observed_at"] is None
    return (
        row["position_observed_at"] == position[0]
        and row["position_source_id"] == position[1]
        and row["position_seq"] == position[2]
    )


def _checkpoint_out(row: sqlite3.Row) -> dict[str, Any]:
    if row["position_observed_at"] is None:
        position = None
    else:
        position = {
            "observed_at": format_event_time(
                parse_event_time(row["position_observed_at"])
            ),
            "source_id": row["position_source_id"],
            "seq": row["position_seq"],
        }
    wm = row["published_watermark"]
    return {
        "session_id": row["session_id"],
        "checkpoint_name": row["checkpoint_name"],
        "position": position,
        "published_watermark": (
            format_event_time(parse_event_time(wm)) if wm else None
        ),
        "created_at": format_event_time(parse_event_time(row["created_at"])),
    }


# ---------------------------------------------------------------------------
# checkpoint lifecycle
# ---------------------------------------------------------------------------

def create_checkpoint(
    conn: sqlite3.Connection,
    session_id: str,
    checkpoint_name: str,
) -> tuple[dict[str, Any], bool]:
    """Create (or idempotently re-create) a stream checkpoint.

    Returns ``(checkpoint, created)`` where ``created`` is ``True`` for a new
    checkpoint and ``False`` for an idempotent same-position retry.

    The stream tail and the published watermark are read together in one
    deferred read transaction, i.e. a single read-only snapshot, so the
    frozen position and watermark always come from the same generation of the
    stream.  An existing checkpoint with the same name is returned untouched
    when the position still matches; a moved stream makes the same name a
    409 conflict.
    """
    with read_transaction(conn):
        session = _load_session(conn, session_id)
        watermark = session["published_watermark"]
        position = _position_of_last_published(conn, session_id)

    created_at = store_event_time(utc_now())
    try:
        with transaction(conn):
            # Re-validate inside the serialised write transaction so the
            # referenced session can never be racing (checkpoints reference
            # sessions via a foreign key).
            _load_session(conn, session_id)
            conn.execute(
                "INSERT INTO stream_checkpoints (session_id, checkpoint_name,"
                " position_observed_at, position_source_id, position_seq,"
                " published_watermark, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    checkpoint_name,
                    position[0] if position is not None else None,
                    position[1] if position is not None else None,
                    position[2] if position is not None else None,
                    watermark,
                    created_at,
                ),
            )
    except sqlite3.IntegrityError:
        # The name already exists for this session.  Decide idempotent retry
        # vs conflict against the stored, immutable position.
        with read_transaction(conn):
            existing = _load_checkpoint(conn, session_id, checkpoint_name)
            if not _same_position(existing, position):
                stored = existing["position_observed_at"]
                raise ConflictError(
                    f"checkpoint {checkpoint_name!r} already exists for session"
                    f" {session_id!r} at a different position (stored tail:"
                    f" {stored!r}, current tail:"
                    f" {position[0] if position is not None else None!r})"
                )
            return _checkpoint_out(existing), False

    with read_transaction(conn):
        row = _load_checkpoint(conn, session_id, checkpoint_name)
        return _checkpoint_out(row), True


def list_checkpoints(
    conn: sqlite3.Connection, session_id: str
) -> list[dict[str, Any]]:
    with read_transaction(conn):
        _load_session(conn, session_id)
        rows = conn.execute(
            "SELECT * FROM stream_checkpoints WHERE session_id = ?"
            " ORDER BY created_at, checkpoint_name",
            (session_id,),
        ).fetchall()
    return [_checkpoint_out(r) for r in rows]


def get_checkpoint(
    conn: sqlite3.Connection, session_id: str, checkpoint_name: str
) -> dict[str, Any]:
    with read_transaction(conn):
        _load_session(conn, session_id)
        row = _load_checkpoint(conn, session_id, checkpoint_name)
        return _checkpoint_out(row)


def delete_checkpoint(
    conn: sqlite3.Connection, session_id: str, checkpoint_name: str
) -> None:
    """Delete one checkpoint; deleting a missing name is a 404."""
    with transaction(conn):
        _load_session(conn, session_id)
        cur = conn.execute(
            "DELETE FROM stream_checkpoints"
            " WHERE session_id = ? AND checkpoint_name = ?",
            (session_id, checkpoint_name),
        )
        if cur.rowcount == 0:
            raise NotFoundError(
                f"checkpoint {checkpoint_name!r} not found for session"
                f" {session_id!r}"
            )


# ---------------------------------------------------------------------------
# resume reads
# ---------------------------------------------------------------------------

def list_from_checkpoint(
    conn: sqlite3.Connection,
    session_id: str,
    checkpoint_name: str,
    limit: int,
    cursor: Optional[str] = None,
) -> dict[str, Any]:
    """Published readings strictly after a checkpoint, keyset paginated.

    The exclusive lower bound is the pagination ``cursor`` while paging, and
    the checkpoint's frozen position on the first page.  The checkpoint row
    and the page are read in the same snapshot, and the frozen bookmark is
    never updated, so subsequent publishes or closing the session do not
    change the checkpoint's meaning.
    """
    with read_transaction(conn):
        session = _load_session(conn, session_id)
        current_wm = session["published_watermark"]
        cp = _load_checkpoint(conn, session_id, checkpoint_name)
        start_after = None
        if cp["position_observed_at"] is not None and not cursor:
            start_after = (
                cp["position_observed_at"],
                cp["position_source_id"],
                cp["position_seq"],
            )
        page, next_cursor = _published_page(
            conn, session_id, limit, start_after=start_after, cursor=cursor
        )
        cp_wm = cp["published_watermark"]

    return {
        "session_id": session_id,
        "checkpoint_name": checkpoint_name,
        "next_cursor": next_cursor,
        "checkpoint_watermark": (
            format_event_time(parse_event_time(cp_wm)) if cp_wm else None
        ),
        "published_watermark": (
            format_event_time(parse_event_time(current_wm)) if current_wm else None
        ),
        "readings": [_reading_out(r) for r in page],
    }
