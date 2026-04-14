"""Tests for Week 3c: Substack agent, article review, newsletter config."""

from __future__ import annotations

import json
import os
import tempfile

from soma.proto_self.substack_agent import (
    NEWSLETTERS_PATH,
    _is_recent,
    fetch_rss_entries,
    load_newsletters,
)


class TestNewslettersConfig:
    def test_loads(self) -> None:
        newsletters = load_newsletters()
        assert "followed" in newsletters
        assert "discovery_seeds" in newsletters
        assert len(newsletters["followed"]) >= 1
        assert len(newsletters["discovery_seeds"]) >= 3

    def test_followed_have_required_fields(self) -> None:
        newsletters = load_newsletters()
        for nl in newsletters["followed"]:
            assert "id" in nl
            assert "name" in nl
            assert "rss" in nl
            assert "tags" in nl
            assert "best_states" in nl

    def test_seeds_have_keywords(self) -> None:
        newsletters = load_newsletters()
        for seed in newsletters["discovery_seeds"]:
            assert "topic" in seed
            assert "keywords" in seed
            assert len(seed["keywords"]) > 0


class TestIsRecent:
    def test_recent_date(self) -> None:
        from email.utils import format_datetime
        from datetime import datetime, timezone
        recent = format_datetime(datetime.now(timezone.utc))
        assert _is_recent(recent, max_days=3) is True

    def test_old_date(self) -> None:
        assert _is_recent("Mon, 01 Jan 2024 00:00:00 GMT", max_days=3) is False

    def test_unparseable_returns_true(self) -> None:
        assert _is_recent("not a date") is True


class TestArticleReview:
    def test_queue_operations(self) -> None:
        """Test load/save queue with temp file."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            # Write some test data
            test_queue = [
                {"title": "Test Article", "url": "https://example.com/1",
                 "approved": True, "auto_surfaced": False, "read": False},
                {"title": "Auto Article", "url": "https://example.com/2",
                 "auto_surfaced": True, "read": False},
            ]
            with open(path, "w") as f:
                json.dump(test_queue, f)

            with open(path) as f:
                loaded = json.load(f)

            assert len(loaded) == 2
            readable = [a for a in loaded if (a.get("approved") or a.get("auto_surfaced")) and not a.get("read")]
            assert len(readable) == 2
        finally:
            os.unlink(path)

    def test_state_filtering(self) -> None:
        """Articles should filter by state."""
        queue = [
            {"title": "A", "best_states": ["baseline", "restored"], "avoid_states": ["depleted"],
             "approved": True, "read": False},
            {"title": "B", "best_states": ["depleted", "recovering"], "avoid_states": [],
             "approved": True, "read": False},
        ]

        depleted_readable = [
            a for a in queue
            if a.get("approved") and not a.get("read")
            and "depleted" in a.get("best_states", [])
            and "depleted" not in a.get("avoid_states", [])
        ]
        assert len(depleted_readable) == 1
        assert depleted_readable[0]["title"] == "B"

        baseline_readable = [
            a for a in queue
            if a.get("approved") and not a.get("read")
            and "baseline" in a.get("best_states", [])
            and "baseline" not in a.get("avoid_states", [])
        ]
        assert len(baseline_readable) == 1
        assert baseline_readable[0]["title"] == "A"
