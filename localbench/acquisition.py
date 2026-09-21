"""Pinned, resumable acquisitions. Importing this module never contacts a server.

Weights are reused in place whenever possible. Only private staging files are
written, and no downloaded executable, installer, or model code is run here.
"""
from contextlib import closing
from pathlib import Path, PurePosixPath
import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from .config import atomic_json

CHUNK_BYTES = 1024 * 1024
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REV = re.compile(r"^[0-9a-f]{40}$")
_PUBLIC_HOSTS = {"github.com", "huggingface.co", "release-assets.githubusercontent.com",
                 "objects.githubusercontent.com", "cdn-lfs.huggingface.co",
                 "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
                 "us.aws.cdn.hf.co", "eu.aws.cdn.hf.co", "us.gcp.cdn.hf.co"}


def _public_url(url, redirect=False):
    value = urllib.parse.urlsplit(url)
    public_metadata_query = (value.hostname == "huggingface.co" and value.path.startswith("/api/models/") and value.query == "blobs=true")
    if (value.scheme != "https" or value.hostname not in _PUBLIC_HOSTS or
            value.port not in (None, 443) or value.username or value.password or
            value.fragment or (value.query and not redirect and not public_metadata_query)):
        raise ValueError("acquisition requires a public HTTPS URL without credentials")
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl, redirect=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_public(request, timeout):
    # Do not inherit credential-bearing proxy/environment configuration.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirect())
    return opener.open(request, timeout=timeout)


def _relative_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("unsafe artifact filename")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("unsafe artifact filename")
    for part in path.parts:
        if ":" in part or part.endswith((".", " ")) or any(ord(c) < 32 for c in part):
            raise ValueError("unsafe Windows artifact filename")
        stem = part.split(".")[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", stem):
            raise ValueError("unsafe Windows artifact filename")
    return path


def _inside(root, name):
    target = root.joinpath(*_relative_name(name).parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes its private directory")
    return target


def _hash(path, state=None, algorithm="sha256", git_blob=False):
    path = Path(path)
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(("blob " + str(path.stat().st_size) + "\0").encode())
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            if state is not None:
                state.check_budget()
            h.update(chunk)
    return h.hexdigest()


def _pin(artifact):
    if type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0:
        raise ValueError("artifact needs an exact positive byte count")
    sha = artifact.get("sha256")
    blob = artifact.get("git_blob_sha1")
    if not (isinstance(sha, str) and _SHA.fullmatch(sha)):
        if not (isinstance(blob, str) and _REV.fullmatch(blob)):
            raise ValueError("artifact needs a pinned SHA256 or Git blob hash")
    _public_url(artifact["url"])


def _verify(path, artifact, state=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        return False
    if artifact.get("sha256"):
        return _hash(path, state) == artifact["sha256"]
    return _hash(path, state, "sha1", True) == artifact["git_blob_sha1"]


def _disk_headroom(config, remaining, disk_usage=shutil.disk_usage, *, destination=None):
    # Test callers may inject disk_usage; production always checks both volumes.
    roots = config.get("acquisition_disk_roots", {"C": "C:\\", "F": "F:\\"})
    from pathlib import PureWindowsPath
    roots=dict(roots)
    model_volume=PureWindowsPath(str(config['paths'].get('new_model_root',''))).drive.rstrip(':').upper()
    target=PureWindowsPath(str(destination)).drive.rstrip(':').upper() if destination is not None else 'F'
    target=target or 'F'
    volumes=list(dict.fromkeys(['C','F']+([model_volume] if model_volume else [])+([target] if target not in ('C','F') else [])))
    for letter in volumes:
        free = disk_usage(roots.get(letter,letter+':\\')).free
        required = config['limits'].get('minimum_free_'+letter.lower()+'_bytes',config['limits']['minimum_free_f_bytes'])
        if free - (remaining if letter == target else 0) < required:
            raise RuntimeError("disk_headroom_" + letter.lower())


def _tree_bytes(root):
    total = 0
    if Path(root).exists():
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not Path(directory, d).is_symlink()]
            for name in files:
                path = Path(directory, name)
                try:
                    if not path.is_symlink():total += path.stat().st_size
                except FileNotFoundError:
                    continue  # Private package/build tools may remove their temporary files mid-scan.
    return total


def _runtime_headroom(config, additional):
    ledger=Path(config["paths"]["artifacts"])/"runtime-container-storage.json"
    docker_bytes=sum(item["accounted_bytes"] for item in json.loads(ledger.read_text(encoding="utf-8"))["images"].values()) if ledger.exists() else 0
    if _tree_bytes(config["paths"]["runtime_root"]) + docker_bytes + additional > config["limits"]["runtime_and_build_soft_cap_bytes"]:
        raise RuntimeError("runtime_soft_cap")


def _acquired(state, artifact_id):
    with state.db:
        state.db.execute("UPDATE weights SET acquired=1 WHERE id=?", (artifact_id,))
    state.snapshot()


def download(config, state, artifact, path, *, weight=False, model_support=False, opener=None,
             disk_usage=shutil.disk_usage):
    """Acquire one immutable artifact with durable partial identity and checks.

    A 200 response to a resumed request safely restarts the same owned staging
    file. A 206 response must describe exactly the requested offset and pinned
    total. No URL, HTTP body, or environment credential is included in errors.
    """
    _pin(artifact)
    path = Path(path)
    owned_root = Path(config["paths"]["new_model_root"] if weight or model_support else config["paths"]["runtime_root"])
    if not path.resolve().is_relative_to(owned_root.resolve()):
        raise ValueError("download destination is outside its private root")
    artifact_id = artifact.get("artifact_id", "sha256:" + artifact.get("sha256", artifact.get("git_blob_sha1", "")))
    if path.exists():
        if not _verify(path, artifact, state):
            raise RuntimeError("completed_artifact_hash_or_size_mismatch")
        return path
    partial = path.with_name(path.name + ".part")
    sidecar = path.with_name(path.name + ".part.json")
    if any(p.is_symlink() or not p.resolve().is_relative_to(owned_root.resolve()) for p in (path, partial, sidecar)):
        raise ValueError("download staging paths must remain inside their private root")
    identity = {k: artifact.get(k) for k in ("bytes", "sha256", "git_blob_sha1", "url")}
    if partial.exists():
        if not sidecar.is_file() or json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("partial_artifact_identity_mismatch")
        if partial.stat().st_size > artifact["bytes"]:
            raise RuntimeError("partial_artifact_oversized")
    if weight:
        if not artifact.get("sha256"):
            raise ValueError("weights require SHA256")
        state.reserve_weight(artifact_id, artifact["bytes"], "download", str(path))
    remaining = artifact["bytes"] - (partial.stat().st_size if partial.exists() else 0)
    _disk_headroom(config, remaining, disk_usage,destination=path)
    if not weight and not model_support:
        _runtime_headroom(config, remaining)
    if remaining == 0 and partial.exists():
        if not _verify(partial, artifact, state):
            # This file is ours, identified by its matching private sidecar.
            partial.unlink()
            raise RuntimeError("download_hash_mismatch")
        partial.replace(path)
        sidecar.unlink(missing_ok=True)
        if weight:
            _acquired(state, artifact_id)
        return path
    state.check_budget()
    state.start_execution("model download" if weight else "pinned runtime download")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(sidecar, identity)
    open_response = opener or _open_public
    attempts = config["limits"].get("retry_transient_attempts", 2) + 1
    for attempt in range(attempts):
        state.check_budget()
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(artifact["url"], headers={"Accept-Encoding": "identity", "User-Agent": "localbench/1"})
        if offset:
            request.add_header("Range", "bytes=" + str(offset) + "-")
        try:
            with closing(open_response(request, timeout=30)) as response:
                status = response.status
                if status == 200:
                    existing_bytes = offset
                    offset = 0
                    expected_body = artifact["bytes"]
                    mode = "wb"
                    _disk_headroom(config, artifact["bytes"] - existing_bytes, disk_usage,destination=path)
                elif status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match:
                        raise RuntimeError("invalid_content_range")
                    first, last, total = map(int, match.groups())
                    if first != offset or last < first or total != artifact["bytes"] or last != total - 1:
                        raise RuntimeError("invalid_content_range")
                    expected_body = last - first + 1
                    mode = "ab"
                else:
                    raise RuntimeError("unexpected_download_status_" + str(status))
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) != expected_body):
                    raise RuntimeError("download_content_length_mismatch")
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding != "identity":
                    raise RuntimeError("unexpected_download_encoding")
                with partial.open(mode) as target:
                    received = 0
                    while True:
                        state.check_budget()
                        _disk_headroom(config, artifact["bytes"] - offset - received, disk_usage,destination=path)
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        if received + len(chunk) > expected_body:
                            raise RuntimeError("download_body_oversized")
                        target.write(chunk)
                        target.flush()
                        received += len(chunk)
                    os.fsync(target.fileno())
                if received != expected_body:
                    raise OSError("download interrupted")
            break
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            # Deliberately discard the exception text, which may include signed
            # redirect URLs or content from an intermediary.
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError("download_http_" + str(exc.code)) from None
            if attempt + 1 == attempts:
                raise RuntimeError("download_transient_attempts_exhausted") from None
    if not _verify(partial, artifact, state):
        partial.unlink()
        raise RuntimeError("download_hash_mismatch")
    partial.replace(path)
    sidecar.unlink(missing_ok=True)
    if weight:
        _acquired(state, artifact_id)
    return path


def safe_extract(config, state, archive, destination, *, disk_usage=shutil.disk_usage):
    """Extract ordinary ZIP files without path escapes, links, or execution."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(archive) as source:
        entries, names, expanded = [], set(), 0
        for entry in source.infolist():
            raw_name = entry.orig_filename
            name = raw_name.rstrip("/") if entry.is_dir() else raw_name
            target = _inside(destination, name)
            key = name.casefold()
            if key in names:
                raise ValueError("duplicate ZIP destination")
            names.add(key)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("ZIP links or special files are forbidden")
            if entry.flag_bits & 1:
                raise ValueError("encrypted ZIP entries are forbidden")
            if not entry.is_dir():
                expanded += entry.file_size
            entries.append((entry, target))
        _disk_headroom(config, expanded, disk_usage)
        _runtime_headroom(config, expanded)
        for entry, target in entries:
            state.check_budget()
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = _inside(destination, str(target.relative_to(destination)).replace("\\", "/") + ".extracting")
            with source.open(entry) as incoming, temporary.open("wb") as outgoing:
                while chunk := incoming.read(CHUNK_BYTES):
                    state.check_budget()
                    _disk_headroom(config, 0, disk_usage)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if temporary.stat().st_size != entry.file_size:
                raise RuntimeError("ZIP extracted size mismatch")
            temporary.replace(target)
            manifest.append({"path": str(target.relative_to(destination)), "bytes": entry.file_size, "sha256": _hash(target, state)})
    return manifest


def acquire_upstream(config, state):
    """Acquire the reviewed b11040 CUDA12.4 binary and adjacent CUDA DLLs."""
    discovery = json.loads((Path(config["paths"]["artifacts"]) / "engine-discovery.json").read_text(encoding="utf-8-sig"))
    engine = next(e for e in discovery["engines"] if e["id"] == "upstream-llama-nightly")
    pin = engine["pin"]
    if pin.get("release_tag") != "b11040" or pin.get("commit") != "5b335f413e4f73b0809c4fe39af894efbcc6a0d2":
        raise ValueError("upstream discovery pin changed; review before acquisition")
    assets = engine["paths"]
    required = {"llama-b11040-bin-win-cuda-12.4-x64.zip": "6e1962aa19742753d868975d7301fce21bc476401d26c5668debc7f0148c112c",
                "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"}
    if len(assets) != 2 or {a["asset"]: a["sha256"] for a in assets} != required:
        raise ValueError("upstream CUDA assets changed; review before acquisition")
    root = Path(config["paths"]["runtime_root"]) / ("upstream-" + pin["commit"])
    manifest_path = root / "engine-manifest.json"
    destination = root / "bin"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("commit") != pin["commit"]:
            raise RuntimeError("existing upstream manifest pin mismatch")
        if all(_verify(_inside(destination, f["path"]), f, state) for f in manifest["files"]):
            binary = _inside(destination, manifest["binary_relative_path"])
            if binary.is_file():
                return {"binary_path": str(binary), "engine_manifest": manifest}
    for asset in assets:
        archive = download(config, state, asset, root / "archives" / asset["asset"])
        safe_extract(config, state, archive, destination)
    binaries = list(destination.rglob("llama-server.exe"))
    if len(binaries) != 1:
        raise RuntimeError("upstream archive must contain exactly one llama-server.exe")
    binary = binaries[0]
    # DLL ZIP layouts can differ from the executable ZIP. Keep the selected
    # runtime together in the owned directory, without executing either asset.
    for dll in list(destination.rglob("*.dll")):
        adjacent = binary.parent / dll.name
        if dll == adjacent:
            continue
        if adjacent.exists():
            if _hash(dll, state) != _hash(adjacent, state):
                raise RuntimeError("conflicting upstream DLL")
        else:
            _runtime_headroom(config, dll.stat().st_size)
            _disk_headroom(config, dll.stat().st_size)
            shutil.copyfile(dll, adjacent)
    files = [{"path": str(p.relative_to(destination)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": _hash(p, state)}
             for p in sorted(destination.rglob("*")) if p.is_file() and not p.name.endswith(".extracting")]
    manifest = {"id": engine["id"], "commit": pin["commit"], "release_tag": pin["release_tag"],
                "source": "https://github.com/ggml-org/llama.cpp", "assets": assets,
                "binary_relative_path": str(binary.relative_to(destination)).replace("\\", "/"),
                "binary_sha256": _hash(binary, state), "files": files, "executed": False}
    atomic_json(manifest_path, manifest)
    state.entity("engine", engine["id"], manifest)
    return {"binary_path": str(binary), "engine_manifest": manifest}


def _inventory_paths(config, candidate):
    inventory_path = Path(config["paths"]["artifacts"]) / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig")) if inventory_path.is_file() else {}
    paths = [Path(row["path"]) for row in inventory.get("models", []) if Path(row["path"]).name == candidate["filename"]]
    for snapshot in inventory.get("hf_snapshots", []):
        if Path(snapshot["snapshot"]).name != candidate["revision"]:
            continue
        paths.extend(Path(row["path"]) for row in snapshot.get("files", []) if Path(row["path"]).name == candidate["filename"])
    root = Path(config["paths"]["installed_model_root"])
    if root.is_dir():
        paths.extend(root.rglob(candidate["filename"]))
    return list(dict.fromkeys(paths))


def _hf_url(candidate, filename):
    if not _REV.fullmatch(candidate.get("revision", "")) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate.get("repo", "")):
        raise ValueError("HF acquisition requires repository and immutable commit")
    _relative_name(filename)
    return "https://huggingface.co/" + candidate["repo"] + "/resolve/" + candidate["revision"] + "/" + urllib.parse.quote(filename, safe="/")


def acquire_gguf(config, state, candidate):
    """Reuse an exact verified installed/cache GGUF, otherwise acquire one file."""
    if not candidate["filename"].endswith(".gguf") or len(_relative_name(candidate["filename"]).parts) != 1:
        raise ValueError("GGUF acquisition requires one root filename")
    artifact = dict(candidate, url=_hf_url(candidate, candidate["filename"]), artifact_id="hf:" + candidate["repo"] + "@" + candidate["revision"] + "/" + candidate["filename"])
    _pin(artifact)
    for path in _inventory_paths(config, candidate):
        if _verify(path, artifact, state):
            result = {"id": candidate["id"], "path": str(path), "sha256": candidate["sha256"], "bytes": candidate["bytes"], "revision": candidate["revision"], "reuse_new_bytes": 0}
            state.entity("model", candidate["id"], result)
            return result
    root = Path(config["paths"]["new_model_root"]) / candidate["repo"].replace("/", "--") / candidate["revision"]
    path = download(config, state, artifact, root / candidate["filename"], weight=True)
    result = {"id": candidate["id"], "path": str(path), "sha256": candidate["sha256"], "bytes": candidate["bytes"], "revision": candidate["revision"], "reuse_new_bytes": candidate["bytes"]}
    state.entity("model", candidate["id"], result)
    return result


def plan_hf_subset(candidate, metadata, index=None):
    """Select pinned root shards and explicit supporting files, never a repo.

    The caller saves pinned API metadata and the verified index before transfer.
    LFS weights require SHA256; Git files retain their Git blob identity.
    """
    if metadata.get("sha") != candidate.get("revision") or not _REV.fullmatch(candidate.get("revision", "")):
        raise ValueError("HF metadata does not match selected immutable revision")
    siblings = {item["rfilename"]: item for item in metadata.get("siblings", [])}
    if index is not None:
        weights = sorted(set(index.get("weight_map", {}).values()))
        if not weights:
            raise ValueError("weight index is empty")
    else:
        weights = candidate.get("root_weight_files") or [candidate.get("filename", "model.safetensors")]
    if candidate.get("root_weight_files") and set(weights) != set(candidate["root_weight_files"]):
        raise ValueError("root weight list differs from index")
    for name in weights:
        if len(_relative_name(name).parts) != 1 or not name.endswith(".safetensors"):
            raise ValueError("only index-referenced root safetensors are eligible")
    required = {"config.json", *weights}
    if index is not None:
        required.add("model.safetensors.index.json")
    missing = required - siblings.keys()
    if missing:
        raise ValueError("missing required pinned snapshot files: " + ",".join(sorted(missing)))
    optional = {"generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja",
                "chat_template.json", "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "quantize_config.json"}
    if not {"tokenizer.json", "tokenizer.model", "vocab.json"} & siblings.keys():
        raise ValueError("pinned snapshot has no supported tokenizer artifacts")
    selected = []
    for name in sorted(required | (optional & siblings.keys())):
        item = siblings[name]
        lfs = item.get("lfs") or {}
        artifact = {"filename": name, "bytes": item.get("size", lfs.get("size")), "url": _hf_url(candidate, name), "weight": name in weights}
        sha = lfs.get("sha256", lfs.get("oid"))
        if isinstance(sha, str) and _SHA.fullmatch(sha):
            artifact["sha256"] = sha
        else:
            artifact["git_blob_sha1"] = item.get("blobId")
        _pin(artifact)
        if artifact["weight"] and not artifact.get("sha256"):
            raise ValueError("safetensors weights need pinned LFS SHA256")
        selected.append(artifact)
    return {"repo": candidate["repo"], "revision": candidate["revision"], "files": selected,
            "weight_bytes": sum(item["bytes"] for item in selected if item["weight"])}

def _read_public(url, state, limit, opener=None):
    _public_url(url)
    state.check_budget()
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "localbench/1"})
    try:
        with closing((opener or _open_public)(request, timeout=30)) as response:
            if response.status != 200 or response.headers.get("Content-Encoding", "identity") != "identity":
                raise RuntimeError("metadata_download_status_or_encoding")
            chunks, count = [], 0
            while chunk := response.read(CHUNK_BYTES):
                state.check_budget()
                count += len(chunk)
                if count > limit:
                    raise RuntimeError("metadata_download_size_limit")
                chunks.append(chunk)
            return b"".join(chunks)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException):
        raise RuntimeError("metadata_download_failed") from None


def _verify_bytes(data, artifact):
    if len(data) != artifact["bytes"]:
        return False
    if artifact.get("sha256"):
        return hashlib.sha256(data).hexdigest() == artifact["sha256"]
    return hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest() == artifact["git_blob_sha1"]


def _metadata_artifact(candidate, item):
    lfs = item.get("lfs") or {}
    artifact = {"filename": item["rfilename"], "bytes": item.get("size", lfs.get("size")),
                "url": _hf_url(candidate, item["rfilename"])}
    sha = lfs.get("sha256", lfs.get("oid"))
    if isinstance(sha, str) and _SHA.fullmatch(sha):
        artifact["sha256"] = sha
    else:
        artifact["git_blob_sha1"] = item.get("blobId")
    _pin(artifact)
    return artifact


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _copy_pinned(config, state, source, target, artifact, artifact_id, disk_usage):
    temporary = target.with_name(target.name + ".copying")
    sidecar = target.with_name(target.name + ".copying.json")
    identity = {k: artifact.get(k) for k in ("bytes", "sha256", "git_blob_sha1")}
    if temporary.exists():
        if not sidecar.is_file() or json.loads(sidecar.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("staging_copy_identity_mismatch")
        if temporary.stat().st_size > artifact["bytes"]:
            raise RuntimeError("staging_copy_oversized")
    if any(p.is_symlink() or not p.resolve().is_relative_to(Path(config["paths"]["new_model_root"]).resolve())
           for p in (target, temporary, sidecar)):
        raise ValueError("staging copy escapes its private root")
    atomic_json(sidecar, identity)
    offset = temporary.stat().st_size if temporary.exists() else 0
    _disk_headroom(config, artifact["bytes"] - offset, disk_usage,destination=target)
    with Path(source).open("rb") as incoming, temporary.open("ab") as outgoing:
        incoming.seek(offset)
        while chunk := incoming.read(CHUNK_BYTES):
            state.check_budget()
            _disk_headroom(config, artifact["bytes"] - outgoing.tell(), disk_usage,destination=target)
            if outgoing.tell() + len(chunk) > artifact["bytes"]:
                raise RuntimeError("staging_copy_source_changed")
            outgoing.write(chunk)
            outgoing.flush()
        os.fsync(outgoing.fileno())
    if not _verify(temporary, artifact, state):
        temporary.unlink()
        raise RuntimeError("staging_copy_hash_mismatch")
    temporary.replace(target)
    sidecar.unlink(missing_ok=True)
    if artifact["weight"]:
        _acquired(state, artifact_id)


def acquire_hf_subset(config, state, candidate, metadata=None, index=None, *,
                      opener=None, disk_usage=shutil.disk_usage):
    """Materialize an explicitly selected pinned checkpoint, with no remote code.

    Metadata can be supplied from saved discovery. Otherwise fetch only the
    immutable HF API record and root index, then SHA/Git-blob verify the index.
    Complete installed snapshots are returned in place. Same-volume hardlinks
    create no physical weight copy; cross-volume copies consume reservations.
    """
    _hf_url(candidate, "config.json")
    artifact_root = Path(config["paths"]["artifacts"])
    evidence_name = candidate["repo"].replace("/", "--") + "-" + candidate["revision"]
    if metadata is None:
        # This one fixed public API query controls file metadata only.
        url = "https://huggingface.co/api/models/" + candidate["repo"] + "/revision/" + candidate["revision"] + "?blobs=true"
        raw = _read_public(url, state, 8 * 1024 * 1024, opener)
        metadata = json.loads(raw)
    if metadata.get("sha") != candidate["revision"]:
        raise ValueError("HF metadata does not match selected immutable revision")
    # Retain only public pin/file/license fields, never HTTP redirect URLs.
    metadata = {k: metadata[k] for k in ("id", "sha", "siblings", "cardData") if k in metadata}
    atomic_json(artifact_root / ("hf-metadata-" + evidence_name + ".json"), metadata)
    index_raw = None
    sibling_map = {item["rfilename"]: item for item in metadata.get("siblings", [])}
    if index is None and "model.safetensors.index.json" in sibling_map:
        index_artifact = _metadata_artifact(candidate, sibling_map["model.safetensors.index.json"])
        saved_index = artifact_root / ("hf-index-" + evidence_name + ".json")
        if saved_index.is_file() and _verify(saved_index, index_artifact, state):
            index_raw = saved_index.read_bytes()
        else:
            index_raw = _read_public(index_artifact["url"], state, 32 * 1024 * 1024, opener)
            if not _verify_bytes(index_raw, index_artifact):
                raise RuntimeError("pinned_weight_index_hash_or_size_mismatch")
            _atomic_bytes(saved_index, index_raw)
        index = json.loads(index_raw)
    plan = plan_hf_subset(candidate, metadata, index)
    atomic_json(artifact_root / ("hf-subset-" + evidence_name + ".json"), plan)
    inventory_path = artifact_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig")) if inventory_path.is_file() else {}
    for snapshot in inventory.get("hf_snapshots", []):
        root = Path(snapshot["snapshot"])
        if root.name == candidate["revision"] and all(_verify(_inside(root, f["filename"]), f, state) for f in plan["files"]):
            result = {"id": candidate["id"], "path": str(root), "repo": candidate["repo"],
                      "revision": candidate["revision"], "weight_bytes": plan["weight_bytes"],
                      "reuse_new_bytes": 0, "files": plan["files"], "remote_code": False}
            state.entity("model", candidate["id"], result)
            return result
    if not state.get_control("harness_verified"):
        raise RuntimeError("new model materialization requires verified harness")
    destination = Path(config["paths"]["new_model_root"]) / candidate["repo"].replace("/", "--") / candidate["revision"]
    destination.mkdir(parents=True, exist_ok=True)
    actions, reserved_bytes = [], 0
    for selected in plan["files"]:
        artifact = dict(selected, artifact_id="hf:" + candidate["repo"] + "@" + candidate["revision"] + "/" + selected["filename"])
        target = _inside(destination, selected["filename"])
        if target.exists():
            if not _verify(target, artifact, state):
                raise RuntimeError("completed_artifact_hash_or_size_mismatch")
            continue
        source = None
        source_candidate = dict(candidate, filename=selected["filename"])
        for existing in _inventory_paths(config, source_candidate):
            if _verify(existing, artifact, state):
                source = existing
                break
        if source is not None:
            # Hardlinks preserve the physical artifact and work without global
            # developer mode or administrator authority on Windows.
            try:
                os.link(source, target)
            except OSError:
                pass
            else:
                if not os.path.samefile(source, target) or not _verify(target, artifact, state):
                    target.unlink()
                    raise RuntimeError("reused_hardlink_hash_mismatch")
                continue
        purpose = "staging_copy" if source is not None else "download"
        if artifact["weight"]:
            state.reserve_weight(artifact["artifact_id"], artifact["bytes"], purpose, str(target))
            reserved_bytes += artifact["bytes"]
        actions.append((artifact, target, source))
    # All new weight reservations and disk checks precede every weight byte.
    _disk_headroom(config, sum(a["bytes"] for a, _, _ in actions), disk_usage,destination=destination)
    for artifact, target, source in actions:
        state.check_budget()
        if source is not None:
            _copy_pinned(config, state, source, target, artifact, artifact["artifact_id"], disk_usage)
        elif index_raw is not None and artifact["filename"] == "model.safetensors.index.json":
            _atomic_bytes(target, index_raw)
        else:
            download(config, state, artifact, target, weight=artifact["weight"],
                     model_support=not artifact["weight"], opener=opener, disk_usage=disk_usage)
    if not all(_verify(_inside(destination, f["filename"]), f, state) for f in plan["files"]):
        raise RuntimeError("materialized_snapshot_verification_failed")
    files = [dict(item, sha256=_hash(_inside(destination, item["filename"]), state)) for item in plan["files"]]
    result = {"id": candidate["id"], "path": str(destination), "repo": candidate["repo"],
              "revision": candidate["revision"], "weight_bytes": plan["weight_bytes"],
              "reuse_new_bytes": reserved_bytes, "files": files, "remote_code": False}
    atomic_json(destination / "localbench-snapshot.json", result)
    state.entity("model", candidate["id"], result)
    return result
