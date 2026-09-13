"""Full-prefix disconnect tests at distinct dynamic phases; simulation only."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"artifacts/teleop_six_hour_20260910/stream_phase_grid_v1"


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for clip,at in (("pico",20),("pico",60),("pico",83),("pico",105),("walk008",11)):
        case=OUTPUT/f"{clip}_disconnect_{at}s"
        if (case/"report.json").exists():
            report=json.loads((case/"report.json").read_text())
        elif case.exists():
            rows.append(dict(case=case.name,error="incomplete prior case preserved"))
            continue
        else:
            command=[sys.executable,"-m","gear_sonic.scripts.evaluate_g1_true23_bfmzero_stream",
                     "--clip",clip,"--scenario","disconnect","--fault-at",str(at),
                     "--timing-environment","shared_cpu_with_training_and_replay", "--output",str(case)]
            with (OUTPUT/f"{case.name}.log").open("w") as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                      check=False,timeout=600)
            if result.returncode:
                rows.append(dict(case=case.name,error=f"process exited{result.returncode}"))
                continue
            report=json.loads((case/"report.json").read_text())
        row=dict(case=case.name,**{k:report.get(k) for k in (
            "controls","requested_controls","mode","physical_failure","protocol_scenario_passed",
            "physical_lifecycle_passed","standing_return_verified","range_excess_max_rad",
            "velocity_ratio_max","effort_ratio_max","scenario_passed")})
        rows.append(row)
        print(json.dumps(row),flush=True)
        (OUTPUT/"progress.json").write_text(json.dumps(rows,indent=2))
    (OUTPUT/"summary.json").write_text(json.dumps(dict(cases=rows,
        all_cases_passed=all(row.get("scenario_passed",False) for row in rows),
        full_body_tracking_qualified=False,hardware_authorized=False),indent=2))


if __name__=="__main__":
    main()
