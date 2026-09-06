"""Deterministic local Python source closure for generalist training lineage.

This parses source without importing it. External packages/native libraries
remain the responsibility of the existing runtime/version manifests. Generic
dynamic import helpers trigger a conservative envs/trl/utils namespace sweep;
script-hosted dynamic classes must be provided as explicit class targets.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

GENERALIST_ROOTS = (
    "gear_sonic/scripts/train_g1_true23_generalist.py",
    "gear_sonic/scripts/train_g1_true23_generalist_curriculum.py",
    "gear_sonic/envs/mjlab/sonic_true23_generalist_curriculum.py",
    "gear_sonic/trl/mjlab/native23_generalist_actor.py",
    "gear_sonic/trl/mjlab/native23_generalist_runner.py",
    "gear_sonic/envs/mjlab/sonic_true23_native_model_actuation.py",
    "gear_sonic/envs/mjlab/sonic_true23_causal_multimotion_v14.py",
    "gear_sonic/utils/g1_true23_native_model_actuation.py",
    "gear_sonic/utils/g1_true23_generalist_source_closure.py",
)
DYNAMIC_CLASS_NAMESPACES = ("gear_sonic.envs", "gear_sonic.trl", "gear_sonic.utils")
_MODULE = re.compile(r"^gear_sonic(?:\.[A-Za-z_]\w*)*$")
_CLASS_TARGET = re.compile(r"^(gear_sonic(?:\.[A-Za-z_]\w*)*):[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


@dataclass(frozen=True)
class SourceClosure:
    files: tuple[Path, ...]
    unresolved_dynamic_imports: tuple[str, ...]
    conservative_namespaces: tuple[str, ...]

    def as_source_files(self, repo_root: Path) -> dict[str, Path]:
        root = repo_root.resolve(strict=True)
        return {path.relative_to(root).as_posix(): path for path in self.files}

    def report(self, repo_root: Path) -> dict:
        return {
            "kind": "g1_true23_generalist_local_ast_source_closure_v1",
            "source_files": list(self.as_source_files(repo_root)),
            "unresolved_dynamic_imports": list(self.unresolved_dynamic_imports),
            "conservative_namespaces": list(self.conservative_namespaces),
            "source_imported_or_executed": False,
            "external_runtime_native_libraries_bound": False,
        }


def collect_local_source_closure(
    repo_root: str | Path,
    roots: Iterable[str | Path],
    *,
    class_targets: Iterable[str] = (),
    dynamic_class_namespaces: tuple[str, ...] = DYNAMIC_CLASS_NAMESPACES,
    maximum_files: int = 4096,
) -> SourceClosure:
    """Bind import AST edges, package initializers and literal class roots.

    ``from package import name`` binds ``package`` and also ``package.name``
    when that is a local module. Function-local/conditional imports are included.
    Package initializers keep their full relative path, avoiding basename-key
    collisions in the old causal source list.
    """
    root = Path(repo_root).resolve(strict=True)
    package_root = root / "gear_sonic"
    if not package_root.is_dir():
        raise ValueError("source closure requires a local gear_sonic package")
    if type(maximum_files) is not int or not 1 <= maximum_files <= 10000:
        raise ValueError("maximum_files must be in 1..10000")
    pending: set[Path] = set()
    visited: set[Path] = set()
    dynamic: set[str] = set()
    swept: set[str] = set()

    def add_path(path: Path) -> None:
        path = path.resolve(strict=True)
        if not path.is_relative_to(package_root) or not path.is_file() or path.suffix != ".py":
            raise ValueError(f"source path must be a Python file inside gear_sonic: {path}")
        if "tests" in path.relative_to(package_root).parts:
            raise ValueError(f"training source closure may not depend on tests: {path}")
        if path not in visited:
            pending.add(path)
        if len(pending | visited) > maximum_files:
            raise ValueError("source closure exceeds maximum_files")

    def module_location(name: str) -> Path | None:
        if not _MODULE.fullmatch(name):
            return None
        base = root.joinpath(*name.split("."))
        initializer = base / "__init__.py"
        if initializer.is_file():
            return initializer
        source = base.with_suffix(".py")
        if source.is_file():
            return source
        return None

    def add_module(name: str, *, required: bool = False) -> bool:
        if not _MODULE.fullmatch(name):
            return False
        parts = name.split(".")
        location = module_location(name)
        namespace = root.joinpath(*parts)
        if location is None and not namespace.is_dir():
            if required:
                raise ValueError(f"local source module is missing: {name}")
            return False
        for length in range(1, len(parts)):
            initializer = root.joinpath(*parts[:length], "__init__.py")
            if initializer.is_file():
                add_path(initializer)
        if location is not None:
            add_path(location)
        return True

    def add_target(value: str, *, required: bool = False) -> None:
        match = _CLASS_TARGET.fullmatch(value)
        if match:
            add_module(match.group(1), required=True)
            return
        if not _MODULE.fullmatch(value):
            if required:
                raise ValueError(f"class target is not a local dotted/colon name: {value}")
            return
        parts = value.split(".")
        while parts:
            if add_module(".".join(parts)):
                return
            parts.pop()
        if required:
            raise ValueError(f"local class target module is missing: {value}")

    def sweep_namespaces() -> None:
        for namespace in dynamic_class_namespaces:
            if not _MODULE.fullmatch(namespace) or namespace == "gear_sonic":
                raise ValueError("dynamic class fallback requires explicit local subnamespaces")
            if namespace in swept:
                continue
            directory = root.joinpath(*namespace.split("."))
            if not directory.is_dir():
                continue
            swept.add(namespace)
            add_module(namespace, required=True)
            for parent, dirs, files in os.walk(directory, followlinks=False):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if d not in {"tests", "__pycache__", ".git"} and not (Path(parent) / d).is_symlink()
                )
                for filename in sorted(files):
                    if filename.endswith(".py"):
                        add_path(Path(parent) / filename)

    for path in roots:
        candidate = Path(path)
        add_path(candidate if candidate.is_absolute() else root / candidate)
    for target in class_targets:
        add_target(target, required=True)
    if not pending:
        raise ValueError("source closure requires at least one local source root")
    total_bytes = 0
    while pending:
        path = min(pending)
        pending.remove(path)
        if path in visited:
            continue
        visited.add(path)
        size = path.stat().st_size
        total_bytes += size
        if size > 4 * 1024 * 1024 or total_bytes > 128 * 1024 * 1024:
            raise ValueError("source closure exceeds Python source size budget")
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(root)
        module_parts = list(relative.with_suffix("").parts)
        is_package = module_parts[-1] == "__init__"
        if is_package:
            module_parts.pop()
        module_name = ".".join(module_parts)
        package_parts = module_parts if is_package else module_parts[:-1]
        add_module(module_name, required=True)
        importer_names = {"import_module", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"importlib", "builtins"}:
                for alias in node.names:
                    if alias.name in {"import_module", "__import__"}:
                        importer_names.add(alias.asname or alias.name)
            if is_package and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                    try:
                        exported = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        exported = ()
                    if isinstance(exported, (tuple, list)):
                        for name in exported:
                            if isinstance(name, str):
                                add_module(f"{module_name}.{name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name, required=alias.name.startswith("gear_sonic."))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package_parts):
                        raise ValueError(f"relative import escapes local package: {path}:{node.lineno}")
                    parent_parts = package_parts[: len(package_parts) - node.level + 1]
                    name = ".".join((*parent_parts, *((node.module or "").split(".") if node.module else ())))
                else:
                    name = node.module or ""
                add_module(name, required=name.startswith("gear_sonic."))
                for alias in node.names:
                    if alias.name != "*":
                        add_module(f"{name}.{alias.name}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                add_target(node.value)
            elif isinstance(node, ast.Call):
                function = node.func
                callee = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else ""
                )
                if callee not in importer_names:
                    continue
                name = (
                    node.args[0].value
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                    else None
                )
                if name is not None and name.startswith("."):
                    package_kw = next(
                        (
                            kw.value.value
                            for kw in node.keywords
                            if kw.arg == "package"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ),
                        None,
                    )
                    if package_kw is None and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                        package_kw = node.args[1].value
                    if isinstance(package_kw, str) and _MODULE.fullmatch(package_kw):
                        dots = len(name) - len(name.lstrip("."))
                        segments = package_kw.split(".")
                        if dots > len(segments):
                            raise ValueError("dynamic relative import escapes local package")
                        name = ".".join((*segments[: len(segments) - dots + 1], name.lstrip(".")))
                    else:
                        name = None
                if name is not None:
                    add_module(name, required=name.startswith("gear_sonic."))
                else:
                    dynamic.add(f"{relative.as_posix()}:{node.lineno}:{callee}")
                    sweep_namespaces()
    return SourceClosure(tuple(sorted(visited)), tuple(sorted(dynamic)), tuple(sorted(swept)))


def generalist_source_files(repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve(strict=True)
    closure = collect_local_source_closure(root, GENERALIST_ROOTS)
    return closure.as_source_files(root)
