"""Read the unchanged independent referee and retain its final private state.

This audit-only adapter adds observation fields immediately before return. It
does not change the referee's native stepping, PD, limits, or failure handling.
No native evaluation occurs on import.
"""
import ast
import hashlib
from pathlib import Path


REFEREE_SHA256 = "0027210cda5a44255debecd7151ecd454e2281a5ff4e3137641d102701fb1330"


def load_oracle_with_endpoint(path):
    path = Path(path)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != REFEREE_SHA256:
        raise ValueError("Independent referee source changed")
    tree = ast.parse(source, filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "inspect_native_segment")
    if not isinstance(function.body[-1], ast.Return):
        raise ValueError("Unexpected referee return structure")
    original_function = ast.dump(function, include_attributes=False)
    additions = ast.parse(
        "if retain_trace:\n"
        "    trace['final_integration'] = _integration_state(model, private)\n"
        "    trace['final_warning_counts'] = private.warning.number.copy()\n"
        "    trace['final_warning_lastinfo'] = private.warning.lastinfo.copy()\n"
    ).body
    function.body[-1:-1] = additions
    # Removing precisely the appended observation recovers the whole function.
    added = function.body.pop(-2)
    if ast.dump(function, include_attributes=False) != original_function:
        raise ValueError("Referee body changed beyond endpoint observation")
    function.body.insert(len(function.body) - 1, added)
    ast.fix_missing_locations(tree)
    namespace = {"__name__": "independent_branch_audit_referee",
                 "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["inspect_native_segment"]
