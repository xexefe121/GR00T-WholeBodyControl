"""Draft only: exactly three current55000 FP32 forwards and three loss gradients.

No optimizer, export, parameter update, native dynamics, or checkpoint selection.
Execution requires a separately frozen request and a concrete reviewed clearance.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
    raise RuntimeError('CUBLAS configuration must precede Torch import.')
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    request_path = HERE/'gradient_request.json'
    clearance_path = HERE/'gradient_clearance.json'
    request, clearance = read(request_path), read(clearance_path)
    request_sha, clearance_sha = sha(request_path), sha(clearance_path)
    if request['budgets'] != dict(forward_calls=3, forward_rows=14110, gradient_calls=3, optimizer_updates=0, native_calls=0, ORT_calls=0):
        raise ValueError('Unexpected fixed budget.')
    if request['schedule_row'] != 0 or request['ordinary_final_step'] != 55000: raise ValueError('Unexpected fixed subject.')
    if not clearance['approved'] or clearance['request_sha256'] != request_sha: raise ValueError('No concrete clearance.')
    if sha(clearance['review_path']) != clearance['review_sha256']: raise ValueError('Changed review.')
    review = read(clearance['review_path'])
    if review[clearance['review_pass_field']] is not True or review['request_sha256'] != request_sha: raise ValueError('Review subject differs.')
    if request['source_sha256'] != sha(__file__): raise ValueError('Executing source differs.')

    def frozen():
        if sha(request_path) != request_sha or sha(clearance_path) != clearance_sha: raise ValueError('Request/clearance changed.')
        for path, digest in request['input_sha256'].items():
            if sha(path) != digest: raise ValueError('Changed frozen input: '+path)
        for role in request['subjects'].values():
            if request['input_sha256'].get(role['path']) != role['sha256'] or sha(role['path']) != role['sha256']:
                raise ValueError('Consumed subject is not pinned.')
        if request['source_sha256'] != sha(__file__): raise ValueError('Source changed.')
    frozen()
    for module in ('direct_model.py', 'direct_data.py', 'direct_contract.py', 'direct_objective.py'):
        path = (Path(request['original_source_directory'])/module).as_posix()
        if request['input_sha256'].get(path) != sha(path): raise ValueError('Unpinned imported source.')
    runtime = request['runtime']
    if Path(sys.executable).resolve() != Path(runtime['python_path']).resolve(): raise ValueError('Wrong Python.')
    if torch.__version__ != runtime['torch_version'] or np.__version__ != runtime['numpy_version'] or torch.version.cuda != runtime['cuda_version']:
        raise ValueError('Wrong runtime version.')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available(): raise ValueError('No selected CUDA device; no fallback.')
    sys.path.insert(0, request['original_source_directory'])
    from direct_model import DirectTarget
    from direct_data import load_data
    from direct_contract import normalized_labels, PHYSICAL_COUNTS
    from direct_objective import nominal_loss, velocity_loss, physical_loss
    dest = HERE/'gradient_results'
    dest.mkdir(exist_ok=False)
    counters = dict(forward_attempted=0, forward_returned=0, forward_rows_attempted=0, forward_rows_returned=0,
                    gradient_attempted=0, gradient_returned=0)
    state = dict(stage='loading', completed=False)
    returned = {}
    gradients = {}
    try:
        fit_request = read(request['subjects']['training_request']['path'])
        data = load_data(fit_request['paths'], request['input_sha256'])
        norm = np.load(request['subjects']['normalization']['path'], allow_pickle=False)
        checkpoint = torch.load(request['subjects']['checkpoint']['path'], map_location='cpu', weights_only=False)
        if checkpoint['ordinary_final_step'] != 55000: raise ValueError('Wrong checkpoint step.')
        for key in ('feature_mean','feature_std','joint_span','runtime_joint_span','default_q','joint_limits','retained_feature_indices'):
            saved = checkpoint[key].detach().cpu().numpy()
            actual = norm[key]
            if saved.shape != actual.shape or saved.dtype != actual.dtype or saved.tobytes() != actual.tobytes():
                raise ValueError('Checkpoint/normalization bytes differ: '+key)
        model = DirectTarget(norm['feature_mean'], norm['feature_std']).to('cuda')
        model.actor.load_state_dict(checkpoint['actor_state'], strict=True)
        model.train()
        parameters = tuple(model.parameters())
        names = [name for name, _ in model.named_parameters()]
        if len(parameters) != 6 or any(p.dtype != torch.float32 for p in parameters): raise ValueError('Parameter contract.')
        for name, value in model.actor.state_dict().items():
            if not torch.equal(value.cpu(), checkpoint['actor_state'][name]): raise ValueError('Restored parameter mismatch.')
        before = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
        tensor = lambda a: torch.from_numpy(np.asarray(a)).to('cuda')
        rows = data['sampled_rows'][0].astype(np.int64)
        axes = data['sampled_axes'][0].astype(np.int64)
        probe_rows = ((rows*23+axes)[:,None]*2+np.arange(2)[None]).reshape(-1)
        mapped = data['center_map'][rows]
        np.savez(dest/'schedule.npz', center_rows=rows, axes=axes, probe_rows=probe_rows, nominal_center_indices=mapped)
        labels = normalized_labels(data['target'], data['default'], data['span'])
        span = data['span'].astype(np.float64)
        velocity_change = (data['velocity_target'].reshape(3057,23,2,23)-data['target'][data['center_map']][:,None,None])/span
        physical_change = (data['physical_target']-data['target'][data['physical_successor']])/span
        predictions = {}
        for label, features, count in (
            ('nominal', data['features'], 9904),
            ('velocity', data['velocity_features'][probe_rows], 1152),
            ('physical', data['physical_features'], 3054)):
            if len(features) != count: raise ValueError('Fixed forward row count.')
            state.update(stage='forward_'+label, active_call_returned=False)
            counters['forward_attempted'] += 1; counters['forward_rows_attempted'] += count
            result = model(tensor(features))
            state['active_call_returned'] = True; counters['forward_returned'] += 1
            torch.cuda.synchronize(); counters['forward_rows_returned'] += count
            actual = result.detach().cpu().numpy().copy()
            returned[label] = actual; np.save(dest/(label+'_prediction.npy'), actual)
            if actual.shape != (count,23) or actual.dtype != np.float32 or not np.isfinite(actual).all(): raise ValueError('Invalid returned prediction.')
            predictions[label] = result
        nl, nc = nominal_loss(predictions['nominal'], tensor(labels), [tensor(ids) for ids in data['cells']])
        vl, vc = velocity_loss(predictions['nominal'], predictions['velocity'], tensor(mapped), tensor(velocity_change[rows,axes]))
        pl, pc = physical_loss(predictions['nominal'], predictions['physical'], tensor(data['physical_successor']), tensor(physical_change),
                               [tensor(ids) for ids in data['physical_cells']], PHYSICAL_COUNTS)
        losses = dict(nominal=nl, velocity=vl, physical=pl)
        np.savez(dest/'losses.npz', nominal=nc.detach().cpu().numpy(), velocity=vc.detach().cpu().numpy(), physical=pc.detach().cpu().numpy())
        for index, (name, loss) in enumerate(losses.items()):
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite loss.')
            state.update(stage='gradient_'+name, active_call_returned=False)
            counters['gradient_attempted'] += 1
            grads = torch.autograd.grad(loss, parameters, retain_graph=index<2, create_graph=False, allow_unused=False)
            state['active_call_returned'] = True; counters['gradient_returned'] += 1
            torch.cuda.synchronize()
            saved = [g.detach().cpu().numpy().copy() for g in grads]
            np.savez(dest/(name+'_gradients.npz'), **dict(zip(names, saved)))
            gradients[name] = np.concatenate([g.reshape(-1).astype(np.float64) for g in saved])
            if not all(np.isfinite(g).all() for g in saved): raise ValueError('Nonfinite returned gradient.')
        norm64 = lambda g: float(np.linalg.norm(g))
        norms = {name: norm64(g) for name, g in gradients.items()}
        cosine = {}
        for left, right in (('nominal','velocity'),('nominal','physical'),('velocity','physical')):
            denominator = norms[left]*norms[right]
            cosine[left+'_'+right] = float(np.dot(gradients[left], gradients[right])/denominator) if denominator else None
        combined = {}
        for weight in (1.,10000.):
            g = gradients['nominal']+weight*gradients['velocity']+gradients['physical']
            combined[str(weight)] = dict(total_norm=norm64(g), velocity_contribution_norm=weight*norms['velocity'],
                velocity_to_nominal_norm=weight*norms['velocity']/norms['nominal'] if norms['nominal'] else None,
                total_cosine_nominal=float(np.dot(g,gradients['nominal'])/(norm64(g)*norms['nominal'])) if norm64(g)*norms['nominal'] else None)
        unchanged = all(np.array_equal(value.detach().cpu().numpy(), before[name]) for name, value in model.state_dict().items())
        if not unchanged or any(p.grad is not None for p in parameters): raise ValueError('Unexpected model/grad-field mutation.')
        frozen()
        state.update(stage='complete', completed=True)
        write(dest/'report.json', dict(passed=True, request_sha256=request_sha, clearance_sha256=clearance_sha,
            counters=counters, losses={k:float(v.detach().cpu()) for k,v in losses.items()}, parameter_names=names,
            parameter_gradient_norms=norms, pairwise_cosines=cosine, fixed_weight_compositions=combined,
            parameter_state_unchanged=unchanged, optimizer_updates=0, native_calls=0, model_scope='original FP32 training model; no FP64 export calls',
            inference_or_generalization_claim=False, selected_weight=None,
            output_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    except BaseException as error:
        write(dest/'failure.json', dict(state=state, counters=counters, error_type=type(error).__name__, error=str(error),
            returned_prediction_names=list(returned), returned_gradient_names=list(gradients), optimizer_updates=0, no_automatic_retry=True,
            output_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
        raise

if __name__ == '__main__': main()
