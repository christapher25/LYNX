"""Allowlisted, detached job runner for LYNX long jobs (backfills, full-history builds).

The lead (orchestrator) uses this instead of starting long jobs by hand:

    python scripts/run_job.py list                  # what may run
    python scripts/run_job.py start fno_features    # launch detached, returns at once
    python scripts/run_job.py status fno_features   # one-line state
    python scripts/run_job.py wait fno_features     # block until it ends; prints a heartbeat

`wait` exits 0 when the job succeeded and 1 when it failed, timed out or died, so
the lead can run it as a background command and be told when the job ends.

Rules this script enforces:
  * only jobs listed in scripts/jobs.json can run (the allowlist is the guardrail);
  * one job at a time (16 GB RAM laptop), unless every running job and the new one
    are marked "parallel_ok";
  * each job runs detached from the terminal and from Claude Code, so closing the
    session does not kill it, and is killed after its "max_hours";
  * stdout/stderr go to timestamped log files; state goes to logs/jobs/<name>.json.

Standard library only. Works on Windows (PyCharm/PowerShell) and Linux.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS_FILE = REPO / "scripts" / "jobs.json"
STATE_DIR = REPO / "logs" / "jobs"
TERMINAL = {"succeeded", "failed", "timeout", "died"}

# Windows process-creation flags (defined here so the module imports on Linux too)
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


# ---------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_jobs() -> dict:
    with open(JOBS_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("jobs", {})


def state_path(name: str) -> Path:
    return STATE_DIR / f"{name}.json"


def read_state(name: str) -> dict | None:
    try:
        return json.loads(state_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_state(name: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = state_path(name).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, state_path(name))  # atomic: a reader never sees half a file


def pid_alive(pid: int | None) -> bool:
    """True if the process exists. Never signals it (os.kill(pid, 0) KILLS on Windows)."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie (exited, not yet reaped) still answers signal 0 on Linux
        with open(f"/proc/{int(pid)}/stat", encoding="ascii", errors="replace") as fh:
            if fh.read().rsplit(")", 1)[-1].split()[0] == "Z":
                return False
    except OSError:
        pass
    return True


def refresh(name: str) -> dict | None:
    """Read the state and mark a 'running' job whose wrapper has vanished as 'died'."""
    st = read_state(name)
    if st and st.get("state") in {"starting", "running"} and not pid_alive(st.get("wrapper_pid")):
        # Give a just-launched wrapper a moment to write its pid.
        if st.get("state") == "starting" and time.time() - st.get("launched_epoch", 0) < 30:
            return st
        if pid_alive(st.get("job_pid")):
            # The job itself is still working; only its supervisor is gone.
            if st.get("note") != "wrapper lost; job still running":
                st["note"] = "wrapper lost; job still running"
                write_state(name, st)
            return st
        st["state"] = "died"
        st["ended_at"] = now_iso()
        st["note"] = "job and wrapper are gone (crash, kill or reboot) without recording an exit code; check the logs"
        st["err_tail"] = tail(st.get("err_log"), 25)
        write_state(name, st)
    return st


def tail(path: str | None, n: int = 15) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    # progress bars write \r without newlines; keep the last few logical lines
    lines = [ln for ln in text.replace("\r", "\n").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def one_line(st: dict) -> str:
    started = st.get("started_at") or st.get("launched_at") or "?"
    bits = [f"{st['name']}: {st['state']}", f"run {st.get('run_id')}", f"started {started}"]
    if st.get("ended_at"):
        bits.append(f"ended {st['ended_at']}")
    if st.get("exit_code") is not None:
        bits.append(f"exit {st['exit_code']}")
    return " | ".join(bits)


# ---------------------------------------------------------------- commands
def cmd_list(_args) -> int:
    for name, job in load_jobs().items():
        print(f"{name:24} max {job.get('max_hours', '?')} h  {job.get('description', '')}")
    return 0


def cmd_start(args) -> int:
    jobs = load_jobs()
    if args.name not in jobs:
        print(f"REFUSED: '{args.name}' is not in scripts/jobs.json. Allowed: {', '.join(jobs) or 'none'}")
        return 3
    job = jobs[args.name]

    # One job at a time unless all involved are parallel_ok.
    for other in jobs:
        st = refresh(other)
        if st and st.get("state") in {"starting", "running"}:
            if other == args.name:
                print(f"REFUSED: {args.name} is already running ({one_line(st)})")
                return 4
            if not (job.get("parallel_ok") and jobs[other].get("parallel_ok")):
                print(f"REFUSED: {other} is still running; one job at a time ({one_line(st)})")
                return 4

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO / job.get("log_dir", "logs/jobs")
    log_dir.mkdir(parents=True, exist_ok=True)
    job_args = job.get("resume_args") if (args.resume and "resume_args" in job) else job.get("args", [])
    state = {
        "name": args.name,
        "run_id": run_id,
        "state": "starting",
        "module": job.get("module"),
        "script": job.get("script"),
        "args": job_args,
        "max_hours": job.get("max_hours", 4),
        "out_log": str(log_dir / f"{args.name}_{run_id}.out.log"),
        "err_log": str(log_dir / f"{args.name}_{run_id}.err.log"),
        "launched_at": now_iso(),
        "launched_epoch": time.time(),
        "wrapper_pid": None,
        "exit_code": None,
    }
    write_state(args.name, state)

    wrapper = [sys.executable, str(Path(__file__).resolve()), "_child", args.name]
    common = dict(cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(wrapper, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **common)
        except OSError:
            proc = subprocess.Popen(wrapper, creationflags=flags, **common)  # job object forbids breakaway
    else:
        proc = subprocess.Popen(wrapper, start_new_session=True, **common)

    state["wrapper_pid"] = proc.pid
    write_state(args.name, state)
    print(f"STARTED {args.name} run {run_id} (wrapper pid {proc.pid})")
    print(f"  logs: {state['out_log']}")
    print(f"        {state['err_log']}")
    print(f"  next: python scripts/run_job.py wait {args.name}   (run it in the background)")
    return 0


def cmd_child(args) -> int:
    """Internal: runs detached, executes the job, records the outcome."""
    st = read_state(args.name) or {}
    job = load_jobs()[args.name]
    if job.get("module"):
        cmd = [sys.executable, "-m", job["module"], *st.get("args", [])]
    else:
        cmd = [sys.executable, str(REPO / job["script"]), *st.get("args", [])]
    st.update(state="running", wrapper_pid=os.getpid(), started_at=now_iso(), command=cmd)
    write_state(args.name, st)

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    extra = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}
    limit = float(st.get("max_hours", 4)) * 3600
    with open(st["out_log"], "ab") as out, open(st["err_log"], "ab") as err:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, env=env, **extra)
        st["job_pid"] = proc.pid
        write_state(args.name, st)
        try:
            code = proc.wait(timeout=limit)
            st["state"] = "succeeded" if code == 0 else "failed"
            st["exit_code"] = code
        except subprocess.TimeoutExpired:
            if os.name == "nt":  # kill the whole tree (a venv python.exe starts a child interpreter)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.kill()
            proc.wait()
            st["state"] = "timeout"
            st["exit_code"] = None
            st["note"] = f"killed after max_hours={st.get('max_hours')}"
    st["ended_at"] = now_iso()
    if st["state"] != "succeeded":
        st["err_tail"] = tail(st["err_log"], 25)
    write_state(args.name, st)
    return 0


def cmd_status(args) -> int:
    st = refresh(args.name)
    if not st:
        print(f"{args.name}: never run")
        return 2
    print(one_line(st))
    if st.get("state") not in TERMINAL:
        last = tail(st.get("err_log"), 1) or tail(st.get("out_log"), 1)
        if last:
            print(f"  last log line: {last}")
    elif st.get("state") != "succeeded":
        print(f"  logs: {st.get('err_log')}")
        print(st.get("err_tail", ""))
    return 0


def cmd_wait(args) -> int:
    if not read_state(args.name):
        print(f"{args.name}: never run")
        return 2
    t0 = time.time()
    last_beat = 0.0
    deadline = t0 + args.timeout_hours * 3600 if args.timeout_hours else None
    while True:
        st = refresh(args.name)
        state = st.get("state")
        if state in TERMINAL:
            print(f"DONE {one_line(st)}")
            if state != "succeeded":
                print(f"  err log: {st.get('err_log')}")
                print(st.get("err_tail", tail(st.get('err_log'), 25)))
                return 1
            return 0
        if time.time() - last_beat >= args.heartbeat_min * 60:
            mins = int((time.time() - t0) / 60)
            last = tail(st.get("err_log"), 1) or tail(st.get("out_log"), 1)
            print(f"[{now_iso()}] {args.name} {state}, waited {mins} min | {last[:160]}", flush=True)
            last_beat = time.time()
        if deadline and time.time() > deadline:
            print(f"WAIT TIMEOUT after {args.timeout_hours} h; job still {state}. Run wait again.")
            return 5
        time.sleep(args.poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description="LYNX allowlisted job runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("start"); p.add_argument("name"); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("status"); p.add_argument("name")
    p = sub.add_parser("wait"); p.add_argument("name")
    p.add_argument("--heartbeat-min", type=float, default=10.0)
    p.add_argument("--poll-sec", type=float, default=15.0)
    p.add_argument("--timeout-hours", type=float, default=0.0, help="0 = wait until the job ends")
    p = sub.add_parser("_child"); p.add_argument("name")
    args = ap.parse_args()
    return {"list": cmd_list, "start": cmd_start, "status": cmd_status,
            "wait": cmd_wait, "_child": cmd_child}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
