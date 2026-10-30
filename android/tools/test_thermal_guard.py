import io
import json
import unittest
from unittest.mock import patch

import thermal_guard as guard


def snapshot(cpu="60.0 °C", gpu="48.0 °C"):
    return {"Children": [
        {"Text": "CPU Package", "SensorId": "/intelcpu/0/power/0", "Value": "25.0 W"},
        {"Text": "CPU Package", "SensorId": "/intelcpu/0/temperature/26", "Value": cpu, "Max": "70.0 °C"},
        {"Text": "GPU Core", "SensorId": "/gpu-nvidia/0/temperature/0", "Value": gpu, "Max": "60.0 °C"},
        {"Text": "GPU Hot Spot", "SensorId": "/gpu-nvidia/0/temperature/2", "Value": "55.0 °C", "Max": "65.0 °C"},
    ]}


class SensorSafetyTests(unittest.TestCase):
    def test_preflight_waits_for_fresh_maxima_before_starting_work(self):
        values = (54, 64, 54, 64)
        with patch.object(guard, "temperatures", side_effect=[guard.SensorUnavailable("Max is pending"), values]), \
             patch.object(guard.time, "monotonic", side_effect=[0, 0.1]), patch.object(guard.time, "sleep") as sleep:
            self.assertEqual(guard.await_temperatures(), values)
            sleep.assert_called_once_with(0.25)

    def test_persistent_preflight_sensor_loss_fails_without_starting_work(self):
        with patch.object(guard, "temperatures", side_effect=guard.SensorUnavailable("Max is pending")), \
             patch.object(guard.time, "monotonic", side_effect=[0, 11]), \
             self.assertRaises(guard.SensorUnavailable):
            guard.await_temperatures()

    def test_console_helpers_are_excluded_but_their_compute_children_are_supervised(self):
        entries = [(8, "conhost.exe", 7), (9, "java.exe", 8), (10, "aapt2.exe", 7),
                   (11, "java.exe", 500), (12, "openconsole.exe", 7)]
        self.assertEqual(guard.supervised_descendants(entries, 7), {9, 10})

    def test_access_denial_on_live_compute_fails_closed(self):
        process = unittest.mock.Mock()
        process.suspend.side_effect = guard.psutil.AccessDenied(9)
        process.is_running.return_value = True
        process.wait.side_effect = guard.psutil.TimeoutExpired(0.25, 9)
        with self.assertRaises(guard.psutil.AccessDenied):
            guard.pause_owned({9: process}, set())

    def test_exit_race_does_not_abort_after_compute_has_disappeared(self):
        process = unittest.mock.Mock()
        process.suspend.side_effect = guard.psutil.AccessDenied(9)
        process.is_running.return_value = False
        paused = set()
        guard.pause_owned({9: process}, paused)
        self.assertEqual(paused, set())

    def test_terminating_process_is_confirmed_by_wait_even_while_pid_is_present(self):
        process = unittest.mock.Mock()
        process.suspend.side_effect = guard.psutil.AccessDenied(9)
        process.is_running.return_value = True
        process.wait.return_value = 0
        paused = set()
        guard.pause_owned({9: process}, paused)
        process.wait.assert_called_once_with(timeout=0.25)
        self.assertEqual(paused, set())

    def test_unverifiable_access_denial_still_fails_closed(self):
        process = unittest.mock.Mock()
        process.suspend.side_effect = guard.psutil.AccessDenied(9)
        process.is_running.return_value = True
        process.wait.side_effect = guard.psutil.AccessDenied(9)
        with self.assertRaises(guard.psutil.AccessDenied):
            guard.pause_owned({9: process}, set())

    def test_finished_process_resume_race_clears_only_owned_pause(self):
        process = unittest.mock.Mock()
        process.resume.side_effect = guard.psutil.AccessDenied(9)
        process.is_running.return_value = True
        process.wait.return_value = 0
        paused = {9}
        guard.resume_owned({9: process}, paused)
        self.assertEqual(paused, set())

    def test_boost_must_be_disabled_on_ac_and_battery(self):
        for ac, dc, accepted in ((0, 0, True), (2, 0, False), (0, 2, False)):
            result = type("Result", (), {"stdout": f"Current AC Power Setting Index: {ac:#x}\nCurrent DC Power Setting Index: {dc:#x}"})()
            with self.subTest(ac=ac, dc=dc), patch.object(guard.subprocess, "run", return_value=result):
                if accepted:
                    guard.verify_boost()
                else:
                    with self.assertRaises(RuntimeError):
                        guard.verify_boost()

    def test_owned_suspend_counts_are_balanced_and_idempotent(self):
        class Process:
            def __init__(self, count): self.count = count
            def suspend(self): self.count += 1
            def resume(self): self.count -= 1
        root, native_child = Process(1), Process(1)
        known, paused = {1: root, 2: native_child}, {1}
        guard.pause_owned(known, paused)
        guard.pause_owned(known, paused)
        self.assertEqual((root.count, native_child.count), (1, 2))
        guard.resume_owned(known, paused)
        self.assertEqual((root.count, native_child.count), (0, 1))
        guard.resume_owned(known, paused)
        self.assertEqual((root.count, native_child.count), (0, 1))

    def test_zero_reading_is_unavailable_and_not_a_safe_temperature(self):
        with self.assertRaises(guard.SensorUnavailable):
            self.read(snapshot(gpu="0.0 °C"))

    def test_ownership_excludes_unrelated_process_trees(self):
        entries = [(9, "java.exe", 8), (11, "python.exe", 500), (8, "cmd.exe", 7), (10, "java.exe", 9)]
        self.assertEqual(guard.descendants(entries, 7), {8, 9, 10})

    def test_inventory_reads_cpu_only_for_external_compute_candidates(self):
        entries = [(10, "java.exe", 7), (11, "python.exe", 500), (12, "msedge.exe", 500)]
        times = type("Times", (), {"user": 1.0, "system": 2.0})()
        with patch.object(guard, "process_snapshot", return_value=entries), patch.object(guard.psutil, "Process") as proc:
            proc.return_value.cpu_times.return_value = times
            self.assertEqual(guard.candidates({10}), {11: 3.0})
            proc.assert_called_once_with(11)

    def read(self, payload):
        response = io.BytesIO(json.dumps(payload).encode())
        with patch.object(guard.urllib.request, "urlopen", return_value=response):
            return guard.temperatures()

    def test_uses_cpu_temperature_not_identically_named_power_sensor(self):
        self.assertEqual(self.read(snapshot()), (60.0, 55.0))

    def test_missing_cpu_or_gpu_fails_closed(self):
        for prefix in ("/intelcpu/", "/gpu-nvidia/"):
            payload = snapshot()
            payload["Children"] = [n for n in payload["Children"] if not n["SensorId"].startswith(prefix)]
            with self.assertRaises(RuntimeError):
                self.read(payload)

    def test_recorded_peaks_catch_spikes_between_samples(self):
        response = io.BytesIO(json.dumps(snapshot()).encode())
        with patch.object(guard.urllib.request, "urlopen", return_value=response):
            self.assertEqual(guard.temperatures(include_peaks=True), (60.0, 55.0, 70.0, 65.0))

    def test_work_budget_shrinks_and_holds_before_pause_threshold(self):
        self.assertEqual(guard.cooling_gap(50, 58), 0.15)
        self.assertEqual(guard.cooling_gap(57, 58), 0.4)
        self.assertEqual(guard.cooling_gap(63, 58), 0.9)
        for cpu, gpu in ((65, 58), (60, 75), (80, 58), (85, 58)):
            self.assertIsNone(guard.cooling_gap(cpu, gpu))

    def test_gpu_hold_stays_below_user_pause_and_shrinks_work_budget(self):
        self.assertFalse(guard.needs_hold(50, 74))
        self.assertTrue(guard.needs_hold(50, 75))
        self.assertEqual(guard.cooling_gap(50, 71), 0.4)
        self.assertEqual(guard.cooling_gap(50, 74), 0.9)
        self.assertLess(guard.GPU_HOLD, guard.PAUSE)

    def test_resume_requires_both_cpu_and_gpu_to_cool(self):
        self.assertTrue(guard.can_resume(59, 69))
        self.assertFalse(guard.can_resume(60, 50))
        self.assertFalse(guard.can_resume(50, 70))

    def test_reset_accepts_actual_get_response_schema_and_rejects_failure(self):
        valid = {"value": 48.0, "min": 48.0, "max": 48.0, "format": "{0:F1} °C"}
        with patch.object(guard, "get_json", side_effect=[snapshot(), valid, valid, valid]):
            guard.reset_peaks()
        with patch.object(guard, "get_json", side_effect=[snapshot(), {"result": "fail"}]), self.assertRaises(RuntimeError):
            guard.reset_peaks()

    def test_invalid_or_non_celsius_readings_fail_closed(self):
        for cpu in ("N/A", "140.0 °F", "NaN °C", "0.0 °C", "200.0 °C"):
            with self.subTest(cpu=cpu), self.assertRaises(RuntimeError):
                self.read(snapshot(cpu=cpu))

    def test_lost_endpoint_does_not_substitute_zero(self):
        with patch.object(guard.urllib.request, "urlopen", side_effect=TimeoutError), self.assertRaises(TimeoutError):
            guard.temperatures()

    def test_both_power_limits_are_required(self):
        for ac, dc, accepted in ((99, 99, True), (100, 99, False), (99, 100, False)):
            result = type("Result", (), {"stdout": f"Current AC Power Setting Index: {ac:#x}\nCurrent DC Power Setting Index: {dc:#x}"})()
            with self.subTest(ac=ac, dc=dc), patch.object(guard.subprocess, "run", return_value=result):
                if accepted:
                    guard.verify_power()
                else:
                    with self.assertRaises(RuntimeError):
                        guard.verify_power()


if __name__ == "__main__":
    unittest.main()
