"""CLI tests for ``python -m farsi_faker``."""

from __future__ import annotations

import io
import json

import pytest

from farsi_faker.__main__ import _emit, _ensure_utf8_stdout, main


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


class TestEmit:
    """The output writer must be a no-op for an empty record list."""

    def test_empty_records_write_nothing_json(self, capsys: pytest.CaptureFixture) -> None:
        _emit([], "json")
        assert capsys.readouterr().out == ""

    def test_empty_records_write_nothing_csv(self, capsys: pytest.CaptureFixture) -> None:
        _emit([], "csv")
        assert capsys.readouterr().out == ""


class TestUtf8Stdout:
    """Persian output must survive a non-UTF-8 (cp1252) console.

    Windows default consoles are cp1252, so printing Persian without
    reconfiguring stdout raises UnicodeEncodeError — the CLI's main use
    case. These tests prove the guard turns that crash into UTF-8 bytes.
    """

    def test_ensure_utf8_reconfigures_cp1252_stream(self, monkeypatch) -> None:
        sink = io.BytesIO()
        # A stdout whose encoding is the Windows default (cp1252).
        stdout = io.TextIOWrapper(sink, encoding="cp1252")
        monkeypatch.setattr("farsi_faker.__main__.sys.stdout", stdout)

        _ensure_utf8_stdout()

        # Persian now encodes without raising and the bytes are UTF-8.
        stdout.write("علی احمدی")
        stdout.flush()
        assert sink.getvalue().decode("utf-8") == "علی احمدی"
        assert stdout.encoding.lower().replace("-", "") == "utf8"

    def test_persian_reaches_stdout_through_ensure(self, monkeypatch) -> None:
        sink = io.BytesIO()
        stdout = io.TextIOWrapper(sink, encoding="cp1252")
        monkeypatch.setattr("farsi_faker.__main__.sys.stdout", stdout)

        _emit([{"name": "علی احمدی"}], "json")
        stdout.flush()

        payload = json.loads(sink.getvalue().decode("utf-8"))
        assert payload[0]["name"] == "علی احمدی"

    def test_ensure_utf8_is_noop_when_no_reconfigure(self, monkeypatch) -> None:
        class _NoReconfigure(io.TextIOWrapper):
            reconfigure = None  # simulate an exotic/unreconfigurable stream

        stdout = _NoReconfigure(io.BytesIO(), encoding="utf-8")
        monkeypatch.setattr("farsi_faker.__main__.sys.stdout", stdout)
        # Must not raise even though the stream cannot be reconfigured.
        _ensure_utf8_stdout()
