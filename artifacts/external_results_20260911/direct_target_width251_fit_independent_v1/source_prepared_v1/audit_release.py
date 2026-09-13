"""Literal single-condition release roles; no file reads or execution."""
from pathlib import Path

def release_paths(base,request):
    base=Path(base);fit=base/'fit';shared=fit/'shared'
    paths=dict(fit_report=fit/'report.json',checkpoint=fit/'student_head.pt',head=fit/'student_head.onnx',
        normalization=shared/'normalization.npz',training_manifest=base/'training_frozen_inputs.json',
        training_request=base/'training_request.json',export_manifest=fit/'output_manifest.json',
        coefficient=Path(request['subjects']['coefficient_source']['path']),
        source_checkpoint=Path(request['subjects']['checkpoint']['path']),
        full_state_generation_request=Path(request['full_state_paths']['request']),
        full_state_generation_report=Path(request['full_state_paths']['report']),
        full_state_data_audit=Path(request['subjects']['full_state_root_audit']['path']),
        full_state_data_owner=Path(request['subjects']['full_state_root_owner']['path']),
        shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json',
        energy_source=Path(request['subjects']['energy_source']['path']),recovery_evidence=shared/'recovery_evidence.json')
    for role in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review','selected_protocol'):
        paths[role]=Path(request['subjects'][role]['path'])
    return paths
