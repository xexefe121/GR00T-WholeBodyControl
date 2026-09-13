"""Independent synthetic-only tests of the proposed split first layer.

No task weights, data, optimizer, ONNX, ORT or native runtime are loaded.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import pytest
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def fixture(batch, device='cpu', dtype=torch.float32, nonzero_context=False):
    g = torch.Generator().manual_seed(21019 + batch)
    make = lambda *shape: torch.randn(*shape, generator=g, dtype=dtype).to(device)
    mean, std = make(1323), make(1323).abs() + .5
    features = make(batch, 1323)
    w0, b0 = make(256, 1323) * .025, make(256) * .01
    if not nonzero_context:
        w0[:, 1000:] = 0
    return features, mean, std, w0, b0, make(256, 256) * .025, make(256) * .01, make(23, 256) * .025, make(23) * .01


def split_first(normalized, weight, bias):
    return (F.linear(normalized[:, :1000].contiguous(), weight[:, :1000].contiguous(), bias)
            + F.linear(normalized[:, 1000:].contiguous(), weight[:, 1000:].contiguous(), None))


@pytest.mark.parametrize('batch', [1, 52, 176, 238, 256])
def test_synthetic_cpu_full_network_initial_bytes(batch):
    x, mean, std, w0, b0, w1, b1, w2, b2 = fixture(batch)
    old = (x[:, :1000].contiguous() - mean[:1000].contiguous()) / std[:1000].contiguous()
    normalized = (x - mean) / std
    assert torch.equal(old, normalized[:, :1000].contiguous())
    a = F.linear(old, w0[:, :1000].contiguous(), b0)
    b = split_first(normalized, w0, b0)
    for w, bias in [(w1, b1), (w2, b2)]:
        a, b = F.linear(F.elu(a), w, bias), F.linear(F.elu(b), w, bias)
    assert a.numpy().tobytes() == b.numpy().tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('batch', [1, 52, 176, 238, 256, 1728, 3054, 9904])
def test_synthetic_cuda_first_layer_initial_bytes(batch):
    x, mean, std, w0, b0, *_ = fixture(batch, 'cuda')
    old = (x[:, :1000].contiguous() - mean[:1000].contiguous()) / std[:1000].contiguous()
    normalized = (x - mean) / std
    a = F.linear(old, w0[:, :1000].contiguous(), b0)
    b = split_first(normalized, w0, b0)
    torch.cuda.synchronize()
    assert a.cpu().numpy().tobytes() == b.cpu().numpy().tobytes()


def test_contiguous_views_keep_both_weight_gradients_and_bias_once():
    x, mean, std, w, b, *_ = fixture(13, dtype=torch.float64, nonzero_context=True)
    z = ((x - mean) / std).requires_grad_()
    w.requires_grad_(); b.requires_grad_()
    upstream = torch.linspace(-.6, .7, 13 * 256, dtype=torch.float64).reshape(13, 256)
    y = split_first(z, w, b)
    (y * upstream).sum().backward()
    assert torch.count_nonzero(w.grad[:, 1000:]) > 0
    torch.testing.assert_close(w.grad, upstream.T @ z.detach(), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(z.grad, upstream @ w.detach(), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(b.grad, upstream.sum(0), rtol=0, atol=0)


def test_blinded_context_columns_get_zero_gradient():
    x, mean, std, w, b, *_ = fixture(17, dtype=torch.float64)
    x[:, 1000:] = mean[1000:]
    z = (x - mean) / std
    w.requires_grad_(); b.requires_grad_()
    split_first(z, w, b).square().sum().backward()
    assert torch.count_nonzero(w.grad[:, 1000:]) == 0
    assert torch.count_nonzero(w.grad[:, :1000]) > 0


def test_nonzero_context_split_and_monolithic_float64_agree_algebraically():
    x, mean, std, w, b, *_ = fixture(19, dtype=torch.float64, nonzero_context=True)
    z = (x - mean) / std
    a = split_first(z, w, b)
    b = F.linear(z, w, b)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_signed_zero_prevents_universal_bit_identity_claim():
    # Exact zero addition can change a sign bit; corpus identity remains measured.
    negative_zero = torch.tensor([-0.], dtype=torch.float32)
    actual = negative_zero + torch.tensor([0.], dtype=torch.float32)
    assert negative_zero.numpy().tobytes() != actual.numpy().tobytes()
