import time

from napari_animation.perf import PerfLogger


def test_interaction_trace_writes_threaded_phase_json(tmp_path):
    import json

    from napari_animation.perf import InteractionTrace

    path = tmp_path / "interaction-trace-test.json"
    trace = InteractionTrace(path)
    with trace.timer("callback"):
        pass
    trace.save()
    events = json.loads(path.read_text())["events"]
    assert len(events) == 1
    assert events[0]["phase"] == "callback"
    assert events[0]["seconds"] >= 0
    assert isinstance(events[0]["thread"], int)


def test_perf_logger_records_phases():
    perf = PerfLogger()
    perf.start()
    with perf.timer("a"):
        time.sleep(0.005)
    with perf.timer("a"):
        time.sleep(0.005)
    with perf.timer("b"):
        time.sleep(0.001)
    report = perf.report()
    assert "a" in report and "b" in report
    # phase "a" ran twice
    assert perf._counts["a"] == 2
    assert perf._totals["a"] >= perf._totals["b"]
    assert "wall-clock" in report


def test_perf_logger_disabled_is_noop():
    perf = PerfLogger(enabled=False)
    with perf.timer("a"):
        pass
    # nothing recorded when disabled
    assert perf._totals == {}
    assert perf.log_report() == perf.report()


def test_nested_frame_timings_and_bounded_history(monkeypatch):
    from napari_animation.perf import render_phase

    clock = iter([0.0, 1.0, 2.0, 3.0, 5.0, 6.0])
    monkeypatch.setattr(
        "napari_animation.perf.time.perf_counter", lambda: next(clock)
    )
    perf = PerfLogger(event_limit=1)
    perf.start()
    with perf.timer("apply", frame=17):
        with render_phase("scene"):
            pass
    perf.stop()
    data = perf.snapshot()
    assert data["wall_seconds"] == 6
    assert data["phases"]["scene"]["total_seconds"] == 1
    assert data["phases"]["apply"]["total_seconds"] == 4
    assert data["events"][0]["frame"] == 17
    assert data["events_dropped"] == 1
    # Stopped reports must not consult a moving clock.
    assert perf.snapshot() == data


def test_concurrent_recording_and_quantiles():
    from concurrent.futures import ThreadPoolExecutor

    perf = PerfLogger(sample_limit=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: perf.record("write", 0.01), range(100)))
    data = perf.snapshot()["phases"]["write"]
    assert data["count"] == 100
    assert abs(data["total_seconds"] - 1) < 1e-9
    assert data["p95_ms"] == 10
    assert data["retained_samples"] == 2


def test_failure_recorded_context_restored_and_json_saved(tmp_path):
    import json

    import pytest

    from napari_animation.perf import render_phase

    perf = PerfLogger()
    perf.start()
    with pytest.raises(RuntimeError):
        with perf.timer("capture", frame=2):
            raise RuntimeError("failure")
    with render_phase("outside_export"):
        pass
    perf.stop()
    path = perf.save(tmp_path / "movie.mp4")
    data = json.loads(path.read_text())
    assert data["phases"]["capture"]["count"] == 1
    assert "outside_export" not in data["phases"]
    assert data["events"][0]["frame"] == 2
    perf.reset()
    assert perf.snapshot()["phases"] == {}


def test_disabled_logger_never_records_or_writes(tmp_path):
    perf = PerfLogger(enabled=False)
    perf.start()
    perf.record("a", 1)
    perf.count("frames_written")
    perf.stop()
    assert perf.save(tmp_path / "movie.mp4") is None
    assert not list(tmp_path.iterdir())
    assert perf.snapshot()["phases"] == {}
    assert perf.snapshot()["counters"] == {}
