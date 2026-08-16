"""Extracting paths from Python by parsing it.

A regex over source text cannot tell a checkpoint from a docstring, an input
from an output, or `MODEL.WEIGHTS` from a filename. It also cannot see through
`os.path.join(ROOT, 'model.pth')`, which is how real code names its files.

Parsing gives all of that: comments and docstrings do not exist in the tree, a
string's meaning comes from the call it sits in, and constants can be resolved
and joined.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Calls whose string arguments are things being *written*.
WRITE_CALLS = {
    "save", "savez", "savez_compressed", "savetxt", "dump", "savefig", "imwrite",
    "to_csv", "to_json", "to_parquet", "write_text", "write_bytes", "save_pretrained",
    "VideoWriter", "makedirs", "mkdir", "write_video", "export", "write_png",
}
# Calls whose string arguments are things being *read*. Kept for confidence, not
# for filtering: a path is a candidate either way.
READ_CALLS = {
    "load", "loadtxt", "imread", "read_image", "read_video", "load_state_dict",
    "from_pretrained", "read_csv", "read_json", "read_parquet", "read_text",
    "load_checkpoint", "loadmat", "open", "get_cfg", "merge_from_file",
}
# Variable names that mean "this is where output goes".
OUTPUT_NAME = re.compile(r"(out|output|save|result|dst|dest|target|log|cache|tmp)", re.IGNORECASE)


@dataclass
class Found:
    path: str
    lineno: int
    is_output: bool


def scan(text: str, filename: str = "<py>") -> Optional[List[Found]]:
    """Every string in ``text`` that could name a file. None if it will not parse."""
    try:
        tree = ast.parse(text, filename=filename)
    except (SyntaxError, ValueError):
        return None

    consts = _module_constants(tree)
    docstrings = _docstring_ids(tree)
    out: List[Found] = []
    _visit(tree, False, consts, docstrings, out)
    return out


MIN_LENGTH = 4


def _visit(node, writing: bool, consts, docstrings: Set[int], out: List[Found]) -> None:
    """Descend, stopping at the largest expression that resolves to a string.

    Stopping matters: `os.path.join(ROOT, "smpl", "SMPL_NEUTRAL.pkl")` is one
    path, and descending into it would also report `ROOT`, `smpl` and the bare
    filename as three more.
    """
    if id(node) in docstrings:
        return

    if isinstance(node, ast.expr):
        value = string_value(node, consts)
        if value is not None:
            # The node resolves, but it may still *be* the write: `os.path.join(
            # out_dir, "ckpt.pt")` names an output whatever encloses it.
            if isinstance(node, ast.Call):
                writing = writing or _is_write_call(node, consts)
            if len(value) >= MIN_LENGTH:
                out.append(Found(value, getattr(node, "lineno", 0), writing))
            return

    # An f-string or concatenation that did not resolve is dynamic. Its literal
    # fragments are not paths: `f"{name}_backbone.pth"` does not require a file
    # called `_backbone.pth`.
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        return

    if isinstance(node, ast.Call):
        writing_here = writing or _is_write_call(node, consts)
        for child in list(node.args) + [kw.value for kw in node.keywords]:
            _visit(child, writing_here, consts, docstrings, out)
        # `open(path, "w").close()` hides the interesting call in the receiver.
        if isinstance(node.func, ast.Attribute):
            _visit(node.func.value, writing, consts, docstrings, out)
        return

    if isinstance(node, ast.Assign):
        # `OUTPUT_PATH = 'results/run.pkl'` is an output by its own name.
        named_output = any(
            isinstance(t, ast.Name) and OUTPUT_NAME.search(t.id) for t in node.targets
        )
        _visit(node.value, writing or named_output, consts, docstrings, out)
        return

    for child in ast.iter_child_nodes(node):
        _visit(child, writing, consts, docstrings, out)


def string_value(node: ast.AST, consts: Dict[str, str]) -> Optional[str]:
    """Resolve a node to a string, following constants, joins and concatenation.

    This is what lets `osp.join(ROOT, 'body_models', 'smpl')` be recognised as
    `dataset/body_models/smpl` — the shape real code uses, and the one a regex
    over string literals can never see. Anything genuinely dynamic returns None.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = string_value(node.left, consts)
        right = string_value(node.right, consts)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):  # f-string: only if fully literal
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call):
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name == "join" and node.args:
            parts = []
            for argument in node.args:
                value = string_value(argument, consts)
                if value is None:
                    return None
                parts.append(value.replace("\\", "/").strip("/"))
            joined = "/".join(p for p in parts if p)
            return joined or None
        if name in ("Path", "PosixPath", "WindowsPath") and node.args:
            return string_value(node.args[0], consts)
    return None


def _module_constants(tree: ast.AST) -> Dict[str, str]:
    """Module-level `NAME = "literal"` bindings, so joins can be resolved."""
    consts: Dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        consts[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and isinstance(node.target, ast.Name):
                consts[node.target.id] = node.value.value
    return consts


def _docstring_ids(tree: ast.AST) -> Set[int]:
    """Node ids of docstrings — prose that happens to contain example paths."""
    ids: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _is_write_call(node: ast.Call, consts: Dict[str, str]) -> bool:
    name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
    if name in WRITE_CALLS:
        return True
    if name == "open":
        # open(path, "w") — the mode is the second positional or a keyword.
        mode = None
        if len(node.args) > 1:
            mode = string_value(node.args[1], consts)
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode = string_value(keyword.value, consts)
        if mode and any(ch in mode for ch in "wax+"):
            return True
    # os.path.join(output_dir, "x.pth") is an output wherever it appears.
    if name == "join" and node.args:
        first = node.args[0]
        label = getattr(first, "id", None) or getattr(first, "attr", None)
        if label and OUTPUT_NAME.search(label):
            return True
    return False
