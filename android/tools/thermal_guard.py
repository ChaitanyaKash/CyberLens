"""Supervise one Windows build using local CPU/GPU sensors and bounded CPU use."""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import psutil

ENDPOINT = "http://127.0.0.1:8085"
PAUSE = 80.0
STOP = 85.0
HOLD = 65.0
RESUME = 60.0
GPU_HOLD = 75.0
GPU_RESUME = 70.0


class SensorUnavailable(RuntimeError):
    pass


def leaves(node):
    yield node
    for child in node.get("Children", []):
        yield from leaves(child)


def get_json(path):
    with urllib.request.urlopen(ENDPOINT + path, timeout=2) as response:
        return json.load(response)


def sensor_nodes():
    sensors = list(leaves(get_json("/data.json")))
    packages = [s for s in sensors if s.get("Text") == "CPU Package"
                and s.get("SensorId", "").startswith("/intelcpu/")
                and "/temperature/" in s.get("SensorId", "")]
    gpu = [s for s in sensors if s.get("SensorId", "").startswith("/gpu-nvidia/")
           and "/temperature/" in s.get("SensorId", "")]
    if len(packages) != 1 or not gpu:
        raise SensorUnavailable("Required CPU Package / NVIDIA temperature sensors missing")
    return packages[0], gpu


def celsius(text):
    match = re.fullmatch(r"\s*(-?\d+(?:[.,]\d+)?)\s*°C\s*", text or "")
    if not match:
        raise SensorUnavailable("Temperature sensor has no valid Celsius value")
    number = float(match[1].replace(",", "."))
    if number <= 0:
        raise SensorUnavailable("Temperature sensor temporarily reports zero or below")
    if not math.isfinite(number) or number >= 125:
        raise RuntimeError("Temperature sensor outside plausible range")
    return number


def temperatures(include_peaks=False):
    cpu, gpu = sensor_nodes()
    def reading(sensor, field):
        try:
            return celsius(sensor.get(field))
        except SensorUnavailable as error:
            raise SensorUnavailable(f"{sensor.get('Text')} {sensor.get('SensorId')} {field}: {sensor.get(field)!r}") from error
    current = reading(cpu, "Value"), max(reading(s, "Value") for s in gpu)
    if not include_peaks:
        return current
    return (*current, reading(cpu, "Max"), max(reading(s, "Max") for s in gpu))


def reset_peaks():
    cpu, gpu = sensor_nodes()
    for sensor in [cpu, *gpu]:
        query = urllib.parse.urlencode({"action": "ResetMinMax", "id": sensor["SensorId"]})
        result = get_json("/Sensor?" + query)
        if not all(key in result for key in ("value", "min", "max", "format")) or "°C" not in result.get("format", ""):
            raise RuntimeError("Could not establish fresh sensor-peak baseline")


def await_temperatures(timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return temperatures(include_peaks=True)
        except (SensorUnavailable, TimeoutError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def verify_power():
    result = subprocess.run(
        ["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", "PROCTHROTTLEMAX"],
        capture_output=True, text=True, check=True,
    )
    values = re.findall(r"Current (?:AC|DC) Power Setting Index:\s*(0x[0-9a-fA-F]+)", result.stdout)
    if len(values) != 2 or any(int(v, 16) != 99 for v in values):
        raise RuntimeError("AC and battery maximum processor state must both be 99%")


def verify_boost():
    result = subprocess.run(
        ["powercfg", "/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", "PERFBOOSTMODE"],
        capture_output=True, text=True, check=True,
    )
    values = re.findall(r"Current (?:AC|DC) Power Setting Index:\s*(0x[0-9a-fA-F]+)", result.stdout)
    if len(values) != 2 or any(int(v, 16) != 0 for v in values):
        raise RuntimeError("AC and battery processor boost must both be Disabled (0)")


def constrain(process, affinity):
    process.cpu_affinity(affinity)
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    class PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x0200, False, process.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        setting = PowerThrottling(1, 1, 0)
        if not kernel.SetProcessInformation(handle, 4, ctypes.byref(setting), ctypes.sizeof(setting)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def process_snapshot():
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    entries = []
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        more = kernel.Process32FirstW(handle, ctypes.byref(entry))
        if not more:
            raise ctypes.WinError(ctypes.get_last_error())
        while more:
            entries.append((entry.pid, entry.name.lower(), entry.parent))
            more = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return entries


def descendants(entries, root):
    owned = {root}
    while True:
        expanded = owned | {pid for pid, _, parent in entries if parent in owned}
        if expanded == owned:
            return owned - {root}
        owned = expanded


def candidates(exclude):
    result = {}
    for pid, name, _ in process_snapshot():
        if pid in exclude:
            continue
        if name.startswith(("java", "python", "cmake", "ninja", "gradle", "emulator", "qemu")):
            try:
                t = psutil.Process(pid).cpu_times()
                result[pid] = t.user + t.system
            except psutil.NoSuchProcess:
                pass
    return result


def supervised_descendants(entries, root):
    owned = descendants(entries, root)
    return {pid for pid, name, _ in entries if pid in owned
            and name not in {"conhost.exe", "openconsole.exe"}}


def busy(before, after):
    return [pid for pid, cpu in after.items() if cpu - before.get(pid, cpu) > 0.1]


def can_resume(cpu, gpu):
    return cpu < RESUME and gpu < GPU_RESUME


def needs_hold(cpu, gpu):
    return cpu >= HOLD or gpu >= GPU_HOLD


def cooling_gap(cpu, gpu):
    if needs_hold(cpu, gpu):
        return None
    if cpu < 55 and gpu < 70:
        return 0.15
    if cpu < 60 and gpu < 73:
        return 0.4
    return 0.9


def confirmed_exited(process):
    if not process.is_running():
        return True
    try:
        process.wait(timeout=0.25)
        return True
    except psutil.NoSuchProcess:
        return True
    except (psutil.TimeoutExpired, psutil.AccessDenied):
        return False


def pause_owned(known, paused):
    for pid, process in list(known.items()):
        if pid not in paused:
            try:
                process.suspend()
                paused.add(pid)
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                if not confirmed_exited(process):
                    raise


def resume_owned(known, paused):
    for pid, process in list(known.items()):
        if pid in paused:
            try:
                process.resume()
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                if not confirmed_exited(process):
                    raise
            paused.discard(pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--log", default="../local_notes/android_2026-10/build-thermal.jsonl")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    verify_power()
    verify_boost()
    cpu, gpu, old_cpu_peak, old_gpu_peak = temperatures(include_peaks=True)
    print(f"CPU Package={cpu:.1f} C; GPU hottest={gpu:.1f} C; AC/DC=99%; hold=CPU {HOLD}/GPU {GPU_HOLD}; pause={PAUSE}; ceiling={STOP}", flush=True)
    if args.probe:
        print(f"Monitor history before this job: CPU max={old_cpu_peak:.1f}; GPU max={old_gpu_peak:.1f}", flush=True)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A build command is required")
    if not can_resume(cpu, gpu):
        raise RuntimeError("Wait for CPU below 60 C and GPU below 70 C before starting")
    own_pid = os.getpid()
    before = candidates({own_pid})
    time.sleep(1)
    if busy(before, candidates({own_pid})):
        raise RuntimeError("Another compute job is active; job not started")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        def event(kind, **kwargs):
            log.write(json.dumps(dict(time=time.time(), event=kind, cpu=cpu, gpu=gpu, **kwargs)) + "\n")
            log.flush()
        event("monitor_baseline", previous_cpu_max=old_cpu_peak, previous_gpu_max=old_gpu_peak)
        reset_peaks()
        cpu, gpu, peak_cpu, peak_gpu = await_temperatures()
        if not can_resume(cpu, gpu):
            raise RuntimeError("Temperature rose during preflight; job not started")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        process = subprocess.Popen(command, env=env, creationflags=0x00000004)
        parent = psutil.Process(process.pid)
        known = {parent.pid: parent}
        owned_paused = {parent.pid}
        affinity = psutil.Process().cpu_affinity()[::2][:2]
        paused = True
        cooling = False
        cooling_since = None
        acknowledged_peak = 0.0
        last_report = 0.0
        last_external_check = time.monotonic()
        external_before = candidates({own_pid, *known})
        external_busy = False
        sensor_wait_since = None

        def discover():
            for pid in supervised_descendants(process_snapshot(), parent.pid):
                if pid not in known:
                    try:
                        p = psutil.Process(pid)
                        known[pid] = p
                        p.suspend()
                        owned_paused.add(pid)
                        constrain(p, affinity)
                    except psutil.NoSuchProcess:
                        continue

        def pause():
            pause_owned(known, owned_paused)

        def cleanup():
            for p in reversed(list(known.values())):
                try:
                    if p.is_running():
                        p.kill()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.wait(timeout=10)

        try:
            constrain(parent, affinity)
            event("start", pid=parent.pid, affinity=affinity, max_duty_percent=40)
            while process.poll() is None:
                discover()
                if process.poll() is not None:
                    break
                try:
                    cpu, gpu, monitor_cpu_peak, monitor_gpu_peak = temperatures(include_peaks=True)
                except (SensorUnavailable, TimeoutError, urllib.error.URLError) as error:
                    pause()
                    if sensor_wait_since is None:
                        sensor_wait_since = time.monotonic()
                        event("sensor_pause", reason=str(error))
                    if time.monotonic() - sensor_wait_since >= 10:
                        raise RuntimeError(f"Sensors unavailable for 10 seconds; job stayed paused: {error}") from error
                    time.sleep(0.5)
                    continue
                if sensor_wait_since is not None:
                    event("sensor_restored")
                    sensor_wait_since = None
                    cooling = True
                    cooling_since = None
                peak_cpu, peak_gpu = max(peak_cpu, monitor_cpu_peak), max(peak_gpu, monitor_gpu_peak)
                hottest = max(cpu, gpu)
                peak = max(peak_cpu, peak_gpu)
                if peak >= STOP:
                    raise RuntimeError(f"85 C ceiling breached: recorded CPU={peak_cpu:.1f} GPU={peak_gpu:.1f}")
                if needs_hold(cpu, gpu) or (peak >= PAUSE and peak > acknowledged_peak):
                    cooling = True
                    cooling_since = None
                    acknowledged_peak = peak
                    event("temperature_hold", peak_cpu=peak_cpu, peak_gpu=peak_gpu)
                if cooling:
                    if can_resume(cpu, gpu):
                        if cooling_since is None:
                            cooling_since = time.monotonic()
                        if time.monotonic() - cooling_since >= 3:
                            cooling = False
                            event("cooled")
                    else:
                        cooling_since = None
                now = time.monotonic()
                if now - last_external_check >= 1:
                    external_after = candidates({own_pid, *known})
                    external_busy = bool(busy(external_before, external_after))
                    external_before = external_after
                    last_external_check = now
                gap = cooling_gap(cpu, gpu)
                if not cooling and gap is not None and not external_busy:
                    resume_owned(known, owned_paused)
                    paused = False
                    time.sleep(0.1)
                    discover()
                    if process.poll() is None:
                        pause()
                    paused = True
                if now - last_report > 30:
                    print(f"Thermal CPU={cpu:.1f} GPU={gpu:.1f}; recorded peak={peak_cpu:.1f}/{peak_gpu:.1f}; cooling={cooling}", flush=True)
                    last_report = now
                time.sleep(gap if gap is not None and not cooling and not external_busy else 0.9)
            event("complete", exit_code=process.returncode, peak_cpu=peak_cpu, peak_gpu=peak_gpu)
            return process.returncode
        except BaseException as error:
            event("abort", reason=str(error), peak_cpu=peak_cpu, peak_gpu=peak_gpu)
            raise
        finally:
            cleanup()


if __name__ == "__main__":
    sys.exit(main())
