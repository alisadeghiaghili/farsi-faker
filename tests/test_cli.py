"""CLI tests for ``python -m farsi_faker``."""

from __future__ import annotations

import json

import pytest

from farsi_faker.__main__ import main


class TestCli:
    """End-to-end CLI behavior."""

    def test_json_single(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["--count", "1", "--seed", "42"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 1
        assert {"name", "first_name", "last_name", "gender"}.issubset(payload[0])

    def test_json_profile(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["--count", "2", "--profile", "--seed", "1"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 2
        assert "national_id" in payload[0]
        assert "mobile" in payload[0]
        assert "email" in payload[0]

    def test_csv_output(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["--count", "3", "--format", "csv", "--seed", "7"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 4  # header + 3
        assert lines[0].startswith("name,")

    def test_gender_filter(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["--count", "5", "--gender", "female", "--seed", "3"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert all(row["gender"] == "female" for row in payload)

    def test_rejects_zero_count(self) -> None:
        with pytest.raises(SystemExit):
            main(["--count", "0"])

    def test_reproducible(self, capsys: pytest.CaptureFixture) -> None:
        main(["--count", "2", "--seed", "99"])
        first = capsys.readouterr().out
        main(["--count", "2", "--seed", "99"])
        second = capsys.readouterr().out
        assert first == second
