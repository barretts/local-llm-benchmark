from pathlib import Path, PureWindowsPath
import re


def safe_source(root, relative, writable=None):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative:
        raise ValueError("invalid relative source path")
    p = PureWindowsPath(relative)
    components = relative.split("/")
    if p.is_absolute() or p.drive or relative.startswith("/") or any(x in ("", ".", "..") for x in components):
        raise ValueError("path escape")
    if components[0] != "src" or not relative.endswith((".py", ".ts")):
        raise ValueError("only fixture source is reachable")
    if any(re.match(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", x) or x.endswith((" ", ".")) for x in components):
        raise ValueError("unsafe Windows path")
    root = Path(root).resolve()
    target = root.joinpath(*components)
    if not target.resolve().is_relative_to(root):
        raise ValueError("symlink/junction escape")
    cursor = target
    while cursor != root:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("symlink/junction source forbidden")
        cursor = cursor.parent
    if writable is not None and relative not in writable:
        raise ValueError("source not writable for this task")
    return target
