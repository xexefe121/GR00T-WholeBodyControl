"""Copy only completed small media, preserving original render receipts."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
VISUAL = Path('C:/Users/camer/.codex/visualizations/2026/09/10/01a08b79-d89d-7d13-bcf8-33de331c0024')
ARCHIVE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_media(source, destination, files):
    assert source.resolve().is_relative_to(VISUAL.resolve())
    # Parent archived completed task folders and retained their workspace paths
    # as NTFS junctions. Permit only that explicitly named task archive as well.
    assert any(destination.resolve().is_relative_to(root.resolve()) for root in (BASE, ARCHIVE))
    destination.mkdir(exist_ok=True)
    checks = []
    for original, copied in files:
        src, dst = source / original, destination / copied
        if dst.exists():
            assert sha(src) == sha(dst), str(dst)
        else:
            shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
        checks.append(dict(source=str(src), workspace_copy=str(dst), sha256=sha(src), bytes=src.stat().st_size))
    receipt = dict(kind='completed_media_workspace_copy', canonical_render_receipt_unchanged=True,
                   temporary_png_frames_not_copied=True, files=checks)
    (destination / 'workspace_copy_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(destination=str(destination), copied_bytes=sum(row['bytes'] for row in checks))))


copy_media(VISUAL / 'mpc_failed_replay_v1', BASE / 'mjbatch_native323_replay_v1/walk002_source3_feedback_v2/visual_comparison_v1', (
    ('full_initialization_and_failed_source.fixed_world.mp4', 'full_initialization_and_failed_source.fixed_world.mp4'),
    ('contact_sheet.fixed_world.png', 'contact_sheet.fixed_world.png'),
    ('failure_boundary.fixed_world.png', 'failure_boundary.fixed_world.png'),
    ('render_receipt.json', 'render_receipt.json'),
    ('renderer_snapshot.py', 'completed_renderer_snapshot.py'),
))
copy_media(VISUAL / 'mpc_h30_clipped_partial_v1', BASE / 'bfm_online_intent_v2/mpc_h30_323_clip1_v1/visual_comparison_v1', (
    ('full_initialization_and_partial_source.fixed_world.mp4', 'full_initialization_and_partial_source.fixed_world.mp4'),
    ('contact_sheet.fixed_world.png', 'contact_sheet.fixed_world.png'),
    ('partial_probe_end.fixed_world.png', 'partial_probe_end.fixed_world.png'),
    ('render_receipt.json', 'render_receipt.json'),
    ('renderer_snapshot.py', 'renderer_snapshot.py'),
))
