"""Authorized one-time graceful stop of PID363 after complete checkpoint800."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import zipfile


PID = 363
ROOT = Path(__file__).resolve().parent
RUN = ROOT / "train3h_v1"
CHECKPOINT = RUN / "residual_00800.pt"
EXPECTED = ["/root/.venvs/g1_true23_mjlab/bin/python", "-m", "gear_sonic.scripts.train_g1_true23_bfm_residual"]
OUTPUT = "artifacts/g1_true23_bfm_residual_20260911_v1/train3h_v1"


def command():
    try:
        return Path(f"/proc/{PID}/cmdline").read_bytes().decode().strip("\0").split("\0")
    except FileNotFoundError:
        return []


deadline = time.monotonic() + 1200
receipt = dict(pid=PID, checkpoint=str(CHECKPOINT), requested_stop_after_update=800)
while time.monotonic() < deadline:
    args = command()
    if not args:
        raise RuntimeError("owned trainer already exited before checkpoint-stop signal")
    if args[:3] != EXPECTED or "--output" not in args or args[args.index("--output")+1] != OUTPUT:
        raise RuntimeError("PID command no longer matches the exact owned trainer")
    if CHECKPOINT.exists():
        try:
            stat = CHECKPOINT.stat()
            metric = json.loads((RUN / "metrics.jsonl").read_text().splitlines()[-1])
            with zipfile.ZipFile(CHECKPOINT) as archive:
                if archive.testzip() is not None or not any(name.endswith("data.pkl") for name in archive.namelist()):
                    raise ValueError("incomplete checkpoint zip")
            time.sleep(2)
            new_stat = CHECKPOINT.stat()
            if (stat.st_size,stat.st_mtime_ns) != (new_stat.st_size,new_stat.st_mtime_ns) or metric["update"] < 800:
                continue
            receipt.update(checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                           checkpoint_bytes=new_stat.st_size, complete_metrics_update_before_signal=metric["update"],
                           signal="SIGINT", signal_sent_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            os.kill(PID,signal.SIGINT)
            print(json.dumps(receipt),flush=True)
            break
        except (zipfile.BadZipFile,ValueError,IndexError,FileNotFoundError):
            pass
    time.sleep(2)
else:
    raise TimeoutError("complete checkpoint800 was not reached within20minutes")

wait_until = time.monotonic()+60
while command() and time.monotonic()<wait_until:
    time.sleep(1)
receipt["process_cmdline_gone"] = not bool(command())
receipt["outcome_exists"] = (RUN/"outcome.json").exists()
receipt["checked_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (ROOT/"authorized_stop_after800_receipt.json").open("x") as stream:
    json.dump(receipt,stream,indent=2)
print(json.dumps(receipt),flush=True)
