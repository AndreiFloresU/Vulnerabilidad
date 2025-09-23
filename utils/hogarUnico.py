from typing import Any


def _norm_id(x: Any) -> str:
    if x is None:
        return "0"
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return "0"
    return s


def make_hogar_id(ced_padre: Any, ced_madre: Any) -> str:
    p = _norm_id(ced_padre)
    m = _norm_id(ced_madre)
    if p == "0" and m == "0":
        return ""  # <- antes devolvías None
    if p == "0":
        return m
    if m == "0":
        return p
    return "|".join(sorted([p, m]))
