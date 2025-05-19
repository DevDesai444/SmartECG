import os
from pathlib import Path
import yaml


def _resolve_env(s):
    # minimal ${oc.env:VAR,default} resolution so configs are portable
    if not isinstance(s, str) or not s.startswith("${oc.env:"):
        return s
    inner = s[len("${oc.env:"):-1]
    if "," in inner:
        var, default = inner.split(",", 1)
    else:
        var, default = inner, ""
    return os.environ.get(var.strip(), default.strip())


def _walk(node):
    if isinstance(node, dict):
        return {k: _walk(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(x) for x in node]
    return _resolve_env(node)


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("defaults", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    return _walk(cfg)
