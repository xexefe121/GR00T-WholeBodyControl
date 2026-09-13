from pathlib import Path
from datetime import datetime, timezone

docs = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911')
snap = Path(__file__).resolve().parent
current = docs/'CURRENT.md'
preserved = snap/'CURRENT_before_1514.md'
assert not preserved.exists()
preserved.write_bytes(current.read_bytes())
text = '''# Current native23 simulation work

Updated 2026-09-11 15:14 UTC. User repeatedly instructed continue; original six-hour cutoff is superseded. Continue simulation autonomously. Real Pico, DDS and robot commands remain outside this stage. No fast full-body controller is qualified.

Repo: Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof. NEW: E:/codex-artifacts/sonic23_teleop_resume_20260911. OLD: E:/codex-artifacts/sonic23_teleop_six_hour_20260910. Full history/contracts/failures in SESSION.md and root_status_snapshots/CURRENT_before_1514.md. Never repeat completed native/model experiments just because older historical notes say pending.

## Current selected work

ONE full58 pure finite-feedback data generation RUNNING: NEW/direct_target_full_state_secants_v1/request.json SHA474d5e4a04f84702b8ada696560df4baf6a3ebfa3b0ffebb0c343d4da11ebde1. Root final review0c22fb1c130d34b153e62d7eeb9880c49e8d87c3ff30cf74e54b3398999da65a binds31 Windows pins, seven sources,1645 pure-runtime pins and launcher724150b85e53fce2dca07c458b60026ddff50788b1d9678fb808fa5cc49b8230. Reviewer owns ONE hidden durable launch at15:11:47UTC, wrapper27504. All3057 nominal centers passed; old-overlap52900/140622 exact at latest snapshot, added0. Actual output generation/. Always read active atomic progress through FILE_SHARE_DELETE helper.

Fixed3057 centers x58 tangent axes x2 signs=354612 signed rows. First all140622 original joint-velocity overlaps, then213990 added35-axis rows. Radii: root position .001m, saved-plan root rotation .01rad, joint position .01rad, root linear velocity .05m/s, root angular velocity .25rad/s, joint velocity .01nativecap. Original pure1000 features, full23x58 teacher K, correction clip+/-.1 then native clip. Zero BFM/head/native/replan/optimizer. Static state checks do not prove contact feasibility or closed-loop recovery.

Root independent saved-data auditor source NEW/direct_target_full_state_data_root_review_v1 prepared, actual audit UNRUN. Original sources preserved source_preserved_v1. Pico found three source gaps; v2 fixes producer counts/call accounting, mandatory center/output/manifest linkage, and nonfinite normalized labels. Nine synthetic tests PASS. Source audit94b9281198f1b62d6b55f573ab11e5d5328093a3bb4869e2e6977f014f55f1f8, mathdf8f1b15d9d83dca92666b6a44d33dd921fdc913f41795cd60ccd6377e53ad6b. Await narrow delta review/completed producer before ONE actual saved audit. Coverage: all354612 states/tangents/labels,140622 exact old overlap,213990 independently rebuilt new features/maps, all367570 intended training rows checked for exact-input/normalized-label conflicts.

Next trainer source preparation selected under NEW/direct_target_full_state_student_v1, expert owns. Actual fit waits qualified full58 data. Fixed10000 updates from ordinary55000 to65000, same1000->256ELU->256ELU->23 model/norm, freshAdamW, cosine1e-4->1e-5, wd1e-5,clip10. Preserve9904 nominal/15equal cells and3054 physical/9cells. Full-state54equal cells (9datasetphase x6tangent groups),16pairs/cell=864pairs/1728 endpoints per update. Finite target-span-normalized endpoint-minus-center feedback loss; no radius division. Measure3 initial gradients once, fix full-state weight=nominal gradient norm/full-state gradient norm; nominal/physical weights1. No adaptive sweep/checkpoint selection. Expected146860000 training rows; complete367570 diagnostic rows, four Torch passes,1437 final FP64 ORT calls. Same-weight FP64 export and original1e-5 preclamp parity remain mandatory.

## Latest controller failure and diagnosis

Ordinary55000 weights with qualified same-weight FP64 execution FAILED actual canonical physical test at c291/sub6/time5.83199999999958: right hip-roll speed -22.398976358281068rad/s against20.2916steps/292attempted controls/42learned calls,250startupactor+250backward,0learnedBFM/source/hold. Tracec5829d6e54bc5791bf9e9409d055496f7e94402c9af8e39ec83cf3b654ac4cae. Owner83b28c282bcc2b05fb122eddd2828548d3b3c5351f701772091fefaa5927cd22,5227pins exact/PIDs absent. Root sole2916step native audit188e3521cf110ef0a27ad1ed77196aa2587082bf078b0bf318a23edb453d36e4 exact; intent fullsource/quiet false. Both cached FP64 audit commands already EXECUTED.

Saved semantics complete64f0b4382d21109420bfbcc2dace856655ac8e4a9d0fbff1d48b39792a784506:5283checks, all292issued/42learned feature/history/prior/output/clamp records exact. First feature departure251, first native target clipping256,34/42 clipped. Failure plot/report NEW/direct_target_fp64_failure_diagnostics_v1/results_v3/report.jsonae8f0e5eee316598a45c3ccbc2bc67848deb7023be727fe902895ddf46fc8c85; plot inspected. Learned proposal p50/p95/max .5592/.910905/3.589ms over42calls; not an independent full-loop timing qualification.

Saved full58 fixed-map decomposition e78567ac2e1fa1eb2c5f4b1ab388a2bca7838aa21a75ae0970d8eb485806b43b: all42nominal maps exact. Other35 tangent dimensions have larger response norm than joint-velocity23 in37/41 departed rows, including3/5 preclip rows. Atc252 root-angular contribution alone .17623rad. These are same-chart frozen-plan linear contributions, not causal percentages or a fresh expert replan.

ONE actual3forward/3gradient diagnostic complete at15? Correct completion14:48:22UTC. Fresh v2 wrapper25152/child27704 exit0; prior wrapper28608 failed before Python due unavailable hidden PowerShell Get-FileHash, preserved0calls.209pins exact. Gradient report41feae732e5b594efdf37720671c7aa85503322f285acbbfc92afc09b6300bf0; owner5ccd53ce54670011b03ee420169aa6781438c7b11424b8a4235fdbee6269e1fa. Root saved Gram audit02cdcacf2ff1313288decbe1617b58ed8888db42add8a850f0245c6fcc7ebbdb completed0newcalls. Nominal/velocity/physical gradient norms .00266348/.000372847/.000521267. Current fixed batch Euclidean common-descent weight interval6.111..224.277; norm-equalizing7.14363 lies inside. Blind10000 velocity weight makes nominal infinitesimal loss uphill. This is not an AdamW, later-batch or full-state guarantee. No further gradient diagnostic selected.

## Independent process-clock component

Native adapterv3 already qualified for replay equivalence:18190expert+3158oldfailedhead=21348actual steps,10full95MB MJB serializations exact. Owner64a89e068eb59f8e2d07200104c42ab887b86f7bcef2dd64790f95c199cf996a. Independent saved audit66e6e1855b3f816b9834f11880847154d878913de708dcea177d1509b48717de complete0native/model.

Fixed shared-memory mailbox original11spawned tests PASS Windows and WSL. Root15synthetic checks9d0abd5e780d4ea0d5f96a9d242b9d70dce6f03b443e2ddc71400419d74f3b51; WSLreportfca89e9dbff60f31f1460a7482b656049ef212fe8eb0026232b04d5e5b932f1a, owner25bf5dfa9a1875a6cbbfbeb18fc1b0de30d9dd92401dfd15c25b6054ddfee7bc. No actual500Hz timing qualification follows.

Pico process-clock source/stub preparation NEW/independent_plant_process_clock_v1/source_draft_v2 has18fake tests PASS and9copied components unchanged. Preparing failed-process-start cleanup fix in fresh preserved version. Proposed one recorded-expert-command1569+continuous250 component benchmark,18190native steps, full291 integration/373-f64 captures, one fixed2msepoch, original deadlines/held/fault semantics. Source/loader/durable supervisor/root saved audit still require review before actual run selection. Recorded commands do not qualify online policy or teleoperation.

## Full simulation acceptance still required

Same fast controller must pass full PICO/walk002/walk003/heldoutwalk008, original source coverage/tracking/native limits, measured quiet final3s plus continuous5s hold, original500Hz plant/50Hz policy deadlines, received-input-only behavior, fault-stop/rearm and perturbations. No crop, retime, limit relaxation, root force or reference editing. Current references include prepared preview and ground-truth root; raw Pico/sensor-only remains unqualified.

Offline PICO6530+250 and walk0021417+250 remain fully physically qualified with slow expert methods; query250 walk0031569+250 also qualified offline. Native replay/video evidence remains available. These do not establish fast online23DOF full-body teleoperation.
'''
text = text.replace('complete at15? Correct completion14:48:22UTC.', 'completed14:48:22UTC.')
current.write_text(text, encoding='utf-8')
with (docs/'SESSION.md').open('a', encoding='utf-8') as f:
    f.write('\n\n## 2026-09-11 15:14 UTC: full-state recovery data selected\n\n')
    f.write(text.split('## Current selected work\n\n',1)[1].split('## Full simulation acceptance still required',1)[0])
print('CURRENT rewritten; prior CURRENT preserved; SESSION appended at '+datetime.now(timezone.utc).isoformat())
