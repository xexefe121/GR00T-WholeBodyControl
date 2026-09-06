"""Explicit IEEE float32 policy for a separate simulator training lineage.

Do not reinterpret an existing TF32 run as an IEEE resume. Keep MJLab's
other backend choices unchanged; guard against precision overrides instead
of silently accepting different arithmetic partway through a run.
"""

from contextlib import contextmanager
from importlib import import_module
import inspect
import json
import os
from pathlib import Path

import torch

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

MJLAB_TORCH_SHA256 = "87e0673c850ae6cd5306c9533c928ddb570396323b904f7bbbab589990bc44b1"
OVERRIDES = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


def environment():
    result = {name: os.environ.get(name) for name in OVERRIDES}
    if any(value not in (None, "0") for value in result.values()):
        raise ValueError("IEEE training rejects TF32 environment overrides")
    return result


def backend_state():
    return dict(
        global_fp32=torch.backends.fp32_precision,
        cuda_matmul_fp32=torch.backends.cuda.matmul.fp32_precision,
        cudnn_fp32=torch.backends.cudnn.fp32_precision,
        cudnn_conv_fp32=torch.backends.cudnn.conv.fp32_precision,
        cudnn_rnn_fp32=torch.backends.cudnn.rnn.fp32_precision,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        default_dtype=str(torch.get_default_dtype()),
        cuda_autocast=torch.is_autocast_enabled("cuda"),
    )


EXPECTED_STATE = dict(
    global_fp32="ieee",
    cuda_matmul_fp32="ieee",
    cudnn_fp32="ieee",
    cudnn_conv_fp32="ieee",
    cudnn_rnn_fp32="ieee",
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    deterministic_algorithms=False,
    default_dtype="torch.float32",
    cuda_autocast=False,
)


def require_ieee(expected_environment):
    if environment() != expected_environment or backend_state() != EXPECTED_STATE:
        raise ValueError("IEEE training precision or backend contract changed")


@contextmanager
def ieee_training_precision():
    backend = import_module("mjlab.utils.torch")
    original = backend.configure_torch_backends
    source = Path(inspect.getfile(original)).resolve(strict=True)
    if file_sha256(source) != MJLAB_TORCH_SHA256:
        raise ValueError("MJLab precision helper changed; revalidate the isolated precision comparison")
    requested_environment = environment()
    before = backend_state()

    def guard():
        require_ieee(requested_environment)

    def configure(allow_tf32=False, deterministic=False):
        if allow_tf32 is not False or deterministic is not False:
            raise ValueError("IEEE launcher cannot enable TF32 or change determinism")
        if environment() != requested_environment:
            raise ValueError("IEEE training environment changed before backend configuration")
        original(allow_tf32=False, deterministic=False)
        # PyTorch 2.9's operator-specific defaults can remain TF32 even when
        # MJLab sets the parent cuDNN precision to IEEE. Set every FP32
        # precision level explicitly; do not alter benchmark/determinism.
        torch.backends.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        guard()

    try:
        backend.configure_torch_backends = configure
        configure()
        yield (
            dict(
                kind="g1_true23_ieee_training_precision_v1",
                requested_state=EXPECTED_STATE.copy(),
                environment_overrides=requested_environment,
                torch_version=str(torch.__version__),
                torch_cuda_version=torch.version.cuda,
                mjlab_helper_path=str(source),
                mjlab_helper_sha256=MJLAB_TORCH_SHA256,
                changed_from_inherited_backend_defaults=[
                    "global_fp32",
                    "cuda_matmul_fp32",
                    "cudnn_fp32",
                    "cudnn_conv_fp32",
                    "cudnn_rnn_fp32",
                ],
                applies_before_anchor_creation_and_rollouts=True,
                existing_tf32_checkpoint_relabelled=False,
                hardware_authorized=False,
                deployment_ready=False,
            ),
            guard,
        )
    finally:
        backend.configure_torch_backends = original
        # Restore precision at each level, plus MJLab's two performance flags.
        # An unrelated caller's dtype/autocast/global determinism is untouched.
        torch.backends.fp32_precision = before["global_fp32"]
        torch.backends.cuda.matmul.fp32_precision = before["cuda_matmul_fp32"]
        torch.backends.cudnn.fp32_precision = before["cudnn_fp32"]
        torch.backends.cudnn.conv.fp32_precision = before["cudnn_conv_fp32"]
        torch.backends.cudnn.rnn.fp32_precision = before["cudnn_rnn_fp32"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]


def guard_algorithm(algorithm, guard):
    guard()
    for model in (algorithm.actor, algorithm.critic):
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("IEEE actor and critic must use float32 parameters")

        def before_forward(module, args):
            guard()

        model.register_forward_pre_hook(before_forward)
    original = algorithm.update

    def checked_update(*args, **kwargs):
        guard()
        result = original(*args, **kwargs)
        guard()
        return result

    algorithm.update = checked_update


def write_runtime(path, document):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("IEEE runtime receipt may not be a symlink")
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("IEEE precision runtime changed on resume")
    else:
        with path.open("x") as stream:
            stream.write(encoded)
