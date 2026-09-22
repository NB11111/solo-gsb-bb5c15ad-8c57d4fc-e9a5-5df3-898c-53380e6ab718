"""End-to-end tests for consumer checkpoints of the published stream."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c


def create_session(client, sources=("a", "b"), lateness=10.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def report(client, sid, readings):
    r = client.post(f"/sessions/{sid}/readings", json={"readings": readings})
    assert r.status_code == 200, r.text
    return r.json()


def seal(client, sid, source, final_observed_at):
    return client.post(
        f"/sessions/{sid}/sources/{source}/seal",
        json={"final_observed_at": final_observed_at},
    )


def create_checkpoint(client, sid, name):
    return client.post(
        f"/sessions/{sid}/checkpoints", json={"checkpoint_name": name}
    )


def resume(client, sid, name, **params):
    return client.get(
        f"/sessions/{sid}/checkpoints/{name}/stream", params=params
    )


def all_resumed(client, sid, name, limit=1000):
    out = []
    cursor = None
    while True:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        body = resume(client, sid, name, **params).json()
        out.extend(body["readings"])
        cursor = body["next_cursor"]
        if not cursor:
            return out


# A zero-lateness, two-source publish progression used by several tests.
# Publishing is *strictly before* the watermark WM = min(source max observed_at):
#   batch 1  a@:10, b@:12   -> WM :10  -> buffered (nothing is strictly earlier)
#   batch 2  a@:13, b@:14   -> WM :13  -> publishes a@:10, b@:12
#   batch 3  a@:15, b@:16   -> WM :15  -> publishes a@:13, b@:14
# Each checkpoint/resume scenario below is derived from these exact timings.
BATCH1 = [
    {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
    {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:12Z"},
]
BATCH2 = [
    {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:13Z"},
    {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:14Z"},
]
BATCH3 = [
    {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T00:00:15Z"},
    {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T00:00:16Z"},
]


class TestCreateCheckpoint:
    def test_checkpoint_on_empty_stream_has_null_position(self, client):
        sid = create_session(client)
        r = create_checkpoint(client, sid, "cp")
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["session_id"] == sid
        assert body["checkpoint_name"] == "cp"
        assert body["position"] is None
        assert body["published_watermark"] is None
        assert body["created_at"]

    def test_checkpoint_records_tail_position_and_watermark_from_one_snapshot(
        self, client
    ):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:30Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:30Z"},
        ])
        # WM = :20 -> only a seq 1 (observed_at :00) is published.
        r = create_checkpoint(client, sid, "cp")
        assert r.status_code == 201
        body = r.json()
        assert body["position"] == {
            "observed_at": "2026-01-01T00:00:00Z",
            "source_id": "a",
            "seq": 1,
        }
        assert body["published_watermark"] == "2026-01-01T00:00:20Z"

    def test_unknown_session_404(self, client):
        assert create_checkpoint(client, "nope", "cp").status_code == 404

    def test_blank_name_422(self, client):
        sid = create_session(client)
        assert create_checkpoint(client, sid, "   ").status_code == 422
        assert client.post(
            f"/sessions/{sid}/checkpoints", json={}
        ).status_code == 422

    def test_name_unique_per_session_but_shareable_across_sessions(self, client):
        s1 = create_session(client, session_id="s1")
        s2 = create_session(client, session_id="s2")
        assert create_checkpoint(client, s1, "cp").status_code == 201
        assert create_checkpoint(client, s2, "cp").status_code == 201


class TestIdempotencyAndConflict:
    def test_same_name_same_position_is_idempotent(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12 (WM :13)
        r1 = create_checkpoint(client, sid, "cp")
        assert r1.status_code == 201
        first = r1.json()
        assert first["position"] == {
            "observed_at": "2026-01-01T00:00:12Z",
            "source_id": "b",
            "seq": 1,
        }
        assert first["published_watermark"] == "2026-01-01T00:00:13Z"
        r2 = create_checkpoint(client, sid, "cp")
        assert r2.status_code == 200
        # The stored checkpoint is returned unchanged (no new row / timestamp).
        assert r2.json() == first

    def test_idempotent_while_stream_empty(self, client):
        sid = create_session(client)
        assert create_checkpoint(client, sid, "cp").status_code == 201
        r = create_checkpoint(client, sid, "cp")
        assert r.status_code == 200
        assert r.json()["position"] is None

    def test_same_name_different_position_conflicts(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12
        first = create_checkpoint(client, sid, "cp").json()
        assert first["position"]["observed_at"] == "2026-01-01T00:00:12Z"
        report(client, sid, BATCH3)  # publishes a@:13, b@:14 -> tail moves
        r = create_checkpoint(client, sid, "cp")
        assert r.status_code == 409
        assert "different position" in r.json()["detail"]
        # The original checkpoint is preserved.
        got = client.get(f"/sessions/{sid}/checkpoints/cp").json()
        assert got["position"] == first["position"]
        assert got["published_watermark"] == first["published_watermark"]

    def test_conflict_also_from_empty_to_nonempty(self, client):
        sid = create_session(client, lateness=0.0)
        assert create_checkpoint(client, sid, "cp").status_code == 201
        report(client, sid, BATCH1)  # still buffered: stream tail is still empty
        assert create_checkpoint(client, sid, "cp").status_code == 200
        report(client, sid, BATCH2)  # now publishes a@:10, b@:12
        assert create_checkpoint(client, sid, "cp").status_code == 409


class TestListGetDelete:
    def test_list_checkpoints_sorted_creation_order(self, client):
        sid = create_session(client)
        create_checkpoint(client, sid, "zeta")
        create_checkpoint(client, sid, "alpha")
        body = client.get(f"/sessions/{sid}/checkpoints").json()
        assert [c["checkpoint_name"] for c in body["checkpoints"]] == [
            "zeta",
            "alpha",
        ]
        assert body["session_id"] == sid

    def test_list_checkpoints_unknown_session_404(self, client):
        assert client.get("/sessions/nope/checkpoints").status_code == 404

    def test_get_checkpoint(self, client):
        sid = create_session(client)
        create_checkpoint(client, sid, "cp")
        r = client.get(f"/sessions/{sid}/checkpoints/cp")
        assert r.status_code == 200
        assert r.json()["checkpoint_name"] == "cp"

    def test_get_unknown_checkpoint_404(self, client):
        sid = create_session(client)
        assert client.get(f"/sessions/{sid}/checkpoints/nope").status_code == 404

    def test_delete_checkpoint(self, client):
        sid = create_session(client)
        create_checkpoint(client, sid, "cp")
        r = client.delete(f"/sessions/{sid}/checkpoints/cp")
        assert r.status_code == 204
        assert r.content == b""
        assert client.get(f"/sessions/{sid}/checkpoints/cp").status_code == 404
        assert client.get(f"/sessions/{sid}/checkpoints").json()["checkpoints"] == []

    def test_delete_unknown_checkpoint_404(self, client):
        sid = create_session(client)
        assert client.delete(f"/sessions/{sid}/checkpoints/nope").status_code == 404

    def test_delete_unknown_session_404(self, client):
        assert client.delete("/sessions/nope/checkpoints/cp").status_code == 404

    def test_name_is_reusable_after_delete(self, client):
        sid = create_session(client, lateness=0.0)
        create_checkpoint(client, sid, "cp")
        client.delete(f"/sessions/{sid}/checkpoints/cp")
        assert create_checkpoint(client, sid, "cp").status_code == 201


class TestResumeFromCheckpoint:
    def test_resume_returns_only_readings_published_after_position(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12
        create_checkpoint(client, sid, "cp")
        report(client, sid, BATCH3)  # publishes a@:13, b@:14
        body = resume(client, sid, "cp", limit=100).json()
        assert [(r["source_id"], r["seq"]) for r in body["readings"]] == [
            ("a", 2),
            ("b", 2),
        ]
        assert body["next_cursor"] is None

    def test_resume_from_empty_stream_position_reads_from_beginning(self, client):
        sid = create_session(client, lateness=0.0)
        create_checkpoint(client, sid, "cp")
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12
        rows = all_resumed(client, sid, "cp")
        assert [(r["source_id"], r["seq"]) for r in rows] == [("a", 1), ("b", 1)]

    def test_resume_first_page_empty_when_nothing_new_published(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12
        create_checkpoint(client, sid, "cp")
        body = resume(client, sid, "cp").json()
        assert body["readings"] == []
        assert body["next_cursor"] is None

    def test_resume_uses_deterministic_order_and_cursor_pagination(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12
        create_checkpoint(client, sid, "cp")
        # Buffer five more readings and seal both sources so the tail flush
        # publishes all of them in deterministic merge order.
        report(client, sid, [
            {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T00:00:20Z"},
            {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T00:00:20Z"},
            {"source_id": "a", "seq": 4, "observed_at": "2026-01-01T00:00:21Z"},
            {"source_id": "b", "seq": 4, "observed_at": "2026-01-01T00:00:22Z"},
            {"source_id": "a", "seq": 5, "observed_at": "2026-01-01T00:00:22Z"},
        ])
        seal(client, sid, "a", "2026-01-01T00:00:22Z")
        seal(client, sid, "b", "2026-01-01T00:00:22Z")

        pages = []
        cursor = None
        seen_cursors = set()
        for _ in range(10):
            params = {"limit": 2}
            if cursor:
                params["cursor"] = cursor
            body = resume(client, sid, "cp", **params).json()
            pages.append(body["readings"])
            cursor = body["next_cursor"]
            if cursor is None:
                break
            assert cursor not in seen_cursors
            seen_cursors.add(cursor)

        # Readings still buffered at checkpoint time are published by the tail
        # flush as well: a@:13/b@:14 (batch 2's own pair never published
        # because nothing arrived strictly before WM :13) plus the five above.
        flattened = [r for page in pages for r in page]
        assert [(r["source_id"], r["seq"]) for r in flattened] == [
            ("a", 2),
            ("b", 2),
            ("a", 3),
            ("b", 3),
            ("a", 4),
            ("a", 5),
            ("b", 4),
        ]
        assert [len(p) for p in pages] == [2, 2, 2, 1]

    def test_resume_bad_cursor_400(self, client):
        sid = create_session(client)
        create_checkpoint(client, sid, "cp")
        r = resume(client, sid, "cp", cursor="%%%not-base64")
        assert r.status_code == 400

    def test_resume_unknown_checkpoint_404(self, client):
        sid = create_session(client)
        assert resume(client, sid, "nope").status_code == 404

    def test_resume_unknown_session_404(self, client):
        assert resume(client, "nope", "cp").status_code == 404

    def test_resume_response_carries_frozen_and_current_watermark(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # WM :13, publishes a@:10, b@:12
        create_checkpoint(client, sid, "cp")
        report(client, sid, BATCH3)  # WM :15, publishes a@:13, b@:14
        body = resume(client, sid, "cp").json()
        # Frozen watermark stays at creation time; current watermark advanced.
        assert body["checkpoint_watermark"] == "2026-01-01T00:00:13Z"
        assert body["published_watermark"] == "2026-01-01T00:00:15Z"

    def test_later_publishes_do_not_change_checkpoint_meaning(self, client):
        sid = create_session(client, lateness=0.0)
        report(client, sid, BATCH1)
        report(client, sid, BATCH2)  # publishes a@:10, b@:12
        frozen = create_checkpoint(client, sid, "cp").json()
        report(client, sid, BATCH3)  # publishes a@:13, b@:14
        # Stored checkpoint is still exactly what was first frozen.
        assert client.get(f"/sessions/{sid}/checkpoints/cp").json() == frozen
        # Resuming multiple times always yields the same suffix.
        first = all_resumed(client, sid, "cp")
        second = all_resumed(client, sid, "cp")
        assert first == second
        assert [(r["source_id"], r["seq"]) for r in first] == [
            ("a", 2),
            ("b", 2),
        ]

    def test_checkpoint_survives_and_resumes_after_session_closed(self, client):
        sid = create_session(client, lateness=10.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:25Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:30Z"},
        ])
        # slowest source is a@:25 -> WM = :15 -> a@:00 published; a@:25 and
        # b@:30 stay buffered.  The checkpoint freezes just that one reading.
        frozen = create_checkpoint(client, sid, "cp").json()
        assert frozen["position"] == {
            "observed_at": "2026-01-01T00:00:00Z",
            "source_id": "a",
            "seq": 1,
        }
        assert frozen["published_watermark"] == "2026-01-01T00:00:15Z"
        seal(client, sid, "a", "2026-01-01T00:00:25Z")
        seal(client, sid, "b", "2026-01-01T00:00:30Z")
        assert client.get(f"/sessions/{sid}").json()["closed"] is True

        # The frozen bookmark did not move when the tail flush happened...
        got = client.get(f"/sessions/{sid}/checkpoints/cp").json()
        assert got["position"] == frozen["position"]
        # ...but resume sees the tail-published remainder, in merge order.
        rows = all_resumed(client, sid, "cp")
        assert [(r["source_id"], r["seq"]) for r in rows] == [("a", 2), ("b", 1)]
        # Re-resuming after close stays stable.
        assert all_resumed(client, sid, "cp") == rows

    def test_resume_after_close_from_empty_position_reads_whole_stream(self, client):
        sid = create_session(client, lateness=10.0)
        create_checkpoint(client, sid, "cp")
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T00:00:00Z"},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:20Z"},
        ])
        seal(client, sid, "a", "2026-01-01T00:00:00Z")
        seal(client, sid, "b", "2026-01-01T00:00:20Z")
        rows = all_resumed(client, sid, "cp")
        assert [(r["source_id"], r["seq"]) for r in rows] == [
            ("a", 1),
            ("b", 1),
        ]


class TestCheckpointMigration:
    def test_legacy_database_is_upgraded_with_checkpoint_table(self, tmp_path):
        import sqlite3

        from app import database
        from app.database import utc_now
        from app.time_utils import store_event_time

        db_path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(database.SCHEMA_V0)
            now = store_event_time(utc_now())
            conn.execute(
                "INSERT INTO sessions (session_id, source_ids,"
                " allowed_lateness_seconds, created_at) VALUES (?, ?, 0, ?)",
                ("legacy-s", '["a","b"]', now),
            )
            conn.commit()
        finally:
            conn.close()

        database.init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            versions = [
                r[0]
                for r in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ]
            assert versions == [1, 2, 3]
            names = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "stream_checkpoints" in names
            # Existing data is untouched.
            assert conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = 'legacy-s'"
            ).fetchone()[0] == 1
        finally:
            conn.close()

        # Re-running init is a no-op (migration 3 recorded exactly once).
        database.init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 3"
            ).fetchone()[0] == 1
        finally:
            conn.close()
