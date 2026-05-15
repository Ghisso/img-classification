#!/usr/bin/env python3
"""Sync generated multi-agent bootstrap files and private AI state with HF buckets."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_BUCKET = "Ghisso/vscode_mounts"
BUCKET_PREFIX = "hf://buckets/"
STATE_INCLUDES = (
    "MEMORY.md",
    "plans/**",
    "explorations/**",
    "session_logs/**",
    "quality_reports/**",
)
BOOTSTRAP_PATHS = (
    ".devcontainer",
    ".claude/agents",
    ".claude/hooks",
    ".claude/instructions",
    ".claude/prompts",
    ".claude/review-profiles",
    ".claude/scripts",
    ".claude/settings.json",
    ".claude/skills",
    ".claude/templates",
    ".claude/MEMORY.md",
    ".claude/plans/README.md",
    ".claude/explorations/README.md",
    ".claude/session_logs/README.md",
    ".claude/quality_reports/README.md",
    ".codex",
    ".github/agents",
    ".github/hooks",
    ".github/instructions",
    ".github/copilot-instructions.md",
    ".mcp.json",
    ".vscode/mcp.json",
    "AGENTS.md",
    "CLAUDE.md",
)


def warn(message: str) -> None:
    print(f"WARNING hf-ai-sync: {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"hf-ai-sync: {message}")


def run_git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()
    return Path.cwd().resolve()


def sanitize_component(value: str) -> str:
    cleaned = value.strip().removesuffix(".git")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned)
    cleaned = cleaned.strip(".-_")
    return cleaned or "unknown"


def split_bucket(value: str) -> tuple[str, str]:
    raw = value.strip().rstrip("/")
    if raw.startswith(BUCKET_PREFIX):
        parts = [part for part in raw.removeprefix(BUCKET_PREFIX).split("/") if part]
        if len(parts) < 2:
            raise ValueError("bucket path must include namespace and bucket name")
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc in {"huggingface.co", "www.huggingface.co"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "buckets":
            prefix_parts = parts[4:] if len(parts) > 3 and parts[3] == "tree" else parts[3:]
            return f"{parts[1]}/{parts[2]}", "/".join(prefix_parts)

    parts = [part for part in raw.split("/") if part]
    if len(parts) < 2:
        raise ValueError("bucket must be <namespace>/<bucket> or <namespace>/<bucket>/<prefix>")
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def devcontainer_sync_config(repo_root: Path) -> tuple[str | None, str | None]:
    config_path = repo_root / ".devcontainer" / "devcontainer.json"
    if not config_path.is_file():
        return None, None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        warn(f"could not read devcontainer sync config: {error}")
        return None, None

    container_env = data.get("containerEnv", {})
    if not isinstance(container_env, dict):
        return None, None
    bucket = container_env.get("HF_AI_SYNC_BUCKET")
    prefix = container_env.get("HF_AI_SYNC_PREFIX")
    return (
        str(bucket).strip() if bucket else None,
        str(prefix).strip() if prefix else None,
    )


def derive_project_prefix(repo_root: Path, explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return explicit_prefix.strip("/")

    origin = run_git(repo_root, "config", "--get", "remote.origin.url")
    if not origin:
        warn("remote.origin.url is not set; using projects/local/<repo-name> prefix.")
        return f"projects/local/{sanitize_component(repo_root.name)}"

    host: str | None = None
    remote_path: str | None = None
    if "://" not in origin:
        match = re.match(r"(?:[^@]+@)?([^:]+):(.+)$", origin)
        if match:
            host = match.group(1)
            remote_path = match.group(2)
    if remote_path is None:
        parsed = urlparse(origin)
        host = parsed.hostname
        remote_path = parsed.path.lstrip("/")

    if not host or not remote_path:
        warn("could not parse remote.origin.url; using projects/local/<repo-name> prefix.")
        return f"projects/local/{sanitize_component(repo_root.name)}"

    parts = [sanitize_component(part) for part in remote_path.split("/") if part]
    if parts:
        parts[-1] = sanitize_component(parts[-1].removesuffix(".git"))
    return "/".join(["projects", sanitize_component(host.lower()), *parts])


def join_prefix(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def remote_uri(bucket_id: str, *parts: str) -> str:
    prefix = join_prefix(*parts)
    return f"{BUCKET_PREFIX}{bucket_id}/{prefix}" if prefix else f"{BUCKET_PREFIX}{bucket_id}"


def token_from_cache() -> str | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_path = hf_home / "token"
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def resolve_token() -> tuple[str | None, str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token, name
    token = token_from_cache()
    if token:
        return token, "huggingface cache"
    return None, "none"


def import_hf_api() -> Any | None:
    try:
        from huggingface_hub import HfApi
    except Exception as error:  # pragma: no cover - depends on container tooling.
        warn(f"huggingface_hub is unavailable ({error}); skipping sync.")
        return None
    return HfApi


def sync_bucket_with_cli(
    args: argparse.Namespace,
    *,
    label: str,
    source: str,
    dest: str,
    token: str | None,
    delete: bool,
    include: tuple[str, ...] | None,
) -> bool:
    hf_command = shutil.which("hf")
    if not hf_command:
        warn("neither huggingface_hub nor the `hf` CLI is available; skipping sync.")
        return False

    command = [hf_command, "buckets", "sync", source, dest]
    command.append("--delete" if delete else "--no-delete")
    for pattern in include or ():
        command.extend(["--include", pattern])
    if token:
        command.extend(["--token", token])
    if args.verbose:
        command.append("--verbose")
    else:
        command.append("--quiet")

    result = subprocess.run(command, text=True, check=False)
    if result.returncode != 0:
        warn(f"{label} failed through hf CLI.")
        return False
    info(f"{label}: completed through hf CLI")
    return True


def should_mock(args: argparse.Namespace) -> bool:
    return bool(args.dry_run or os.environ.get("HF_AI_SYNC_MOCK") == "1")


def summarize(label: str, plan: Any) -> None:
    summary = plan.summary()
    info(
        f"{label}: {summary.get('uploads', 0)} uploads, "
        f"{summary.get('downloads', 0)} downloads, "
        f"{summary.get('deletes', 0)} deletes, "
        f"{summary.get('skips', 0)} skips, "
        f"{summary.get('total_size', 0)} bytes"
    )


def sync_bucket(
    args: argparse.Namespace,
    *,
    label: str,
    source: str,
    dest: str,
    token: str | None,
    delete: bool = False,
    include: tuple[str, ...] | None = None,
) -> bool:
    if should_mock(args):
        info(f"dry-run {label}: {source} -> {dest} delete={delete} include={list(include or ())}")
        return True

    if not token:
        warn("no HF token found; run `hf auth login` or set HF_TOKEN. Skipping sync.")
        return False

    HfApi = import_hf_api()
    if HfApi is None:
        return sync_bucket_with_cli(
            args,
            label=label,
            source=source,
            dest=dest,
            token=token,
            delete=delete,
            include=include,
        )

    try:
        plan = HfApi().sync_bucket(
            source=source,
            dest=dest,
            delete=delete,
            include=list(include) if include else None,
            dry_run=False,
            verbose=args.verbose,
            quiet=not args.verbose,
            token=token,
        )
    except Exception as error:  # pragma: no cover - network and HF account dependent.
        warn(f"{label} failed: {error}")
        return False

    summarize(label, plan)
    return True


def chmod_runtime_scripts(repo_root: Path) -> None:
    for pattern in (".claude/hooks/scripts/*.sh", ".devcontainer/*.sh"):
        for path in repo_root.glob(pattern):
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o111)


def copy_bootstrap_path(repo_root: Path, stage_root: Path, relative_path: str) -> bool:
    source = repo_root / relative_path
    if not source.exists():
        return False

    destination = stage_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return True


def upload_bootstrap(args: argparse.Namespace, repo_root: Path, bucket_id: str, prefix: str, token: str | None) -> bool:
    with tempfile.TemporaryDirectory(prefix="hf-ai-bootstrap-") as temp_dir_name:
        stage_root = Path(temp_dir_name)
        copied = [path for path in BOOTSTRAP_PATHS if copy_bootstrap_path(repo_root, stage_root, path)]
        if not copied:
            warn("no generated AI bootstrap files found to upload.")
            return False

        return sync_bucket(
            args,
            label="upload-bootstrap",
            source=str(stage_root),
            dest=remote_uri(bucket_id, prefix, "bootstrap"),
            token=token,
            delete=True,
        )


def pull_bootstrap(args: argparse.Namespace, repo_root: Path, bucket_id: str, prefix: str, token: str | None) -> bool:
    ok = sync_bucket(
        args,
        label="pull-bootstrap",
        source=remote_uri(bucket_id, prefix, "bootstrap"),
        dest=str(repo_root),
        token=token,
        delete=False,
    )
    chmod_runtime_scripts(repo_root)
    return ok


def pull_state(args: argparse.Namespace, repo_root: Path, bucket_id: str, prefix: str, token: str | None) -> bool:
    claude_root = repo_root / ".claude"
    if not should_mock(args) and token:
        claude_root.mkdir(parents=True, exist_ok=True)
    ok = sync_bucket(
        args,
        label="pull-state",
        source=remote_uri(bucket_id, prefix, "state", ".claude"),
        dest=str(claude_root),
        token=token,
        delete=False,
        include=STATE_INCLUDES,
    )
    chmod_runtime_scripts(repo_root)
    return ok


def push_state(args: argparse.Namespace, repo_root: Path, bucket_id: str, prefix: str, token: str | None) -> bool:
    claude_root = repo_root / ".claude"
    if not claude_root.is_dir() and not should_mock(args):
        warn(".claude is missing; nothing to push.")
        return False
    return sync_bucket(
        args,
        label="push-state",
        source=str(claude_root),
        dest=remote_uri(bucket_id, prefix, "state", ".claude"),
        token=token,
        delete=False,
        include=STATE_INCLUDES,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("pull", "pull-bootstrap", "pull-state", "push", "push-state", "upload-bootstrap", "status"),
        help="Sync operation to run.",
    )
    parser.add_argument("--repo-root", help="Repository root. Defaults to git rev-parse or cwd.")
    parser.add_argument(
        "--bucket",
        default=None,
        help="HF bucket id, bucket URL, or hf://buckets path.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Override the derived per-project prefix.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned sync operations without HF access.")
    parser.add_argument("--verbose", action="store_true", help="Show verbose Hugging Face sync output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    devcontainer_bucket, devcontainer_prefix = devcontainer_sync_config(repo_root)
    configured_bucket = (
        args.bucket
        or os.environ.get("HF_AI_SYNC_BUCKET")
        or devcontainer_bucket
        or DEFAULT_BUCKET
    )
    configured_prefix = (
        args.prefix
        if args.prefix is not None
        else os.environ.get("HF_AI_SYNC_PREFIX") or devcontainer_prefix
    )
    try:
        bucket_id, bucket_base_prefix = split_bucket(configured_bucket)
    except ValueError as error:
        warn(str(error))
        return 0

    if configured_prefix:
        project_prefix = configured_prefix.strip("/")
        prefix = join_prefix(bucket_base_prefix, project_prefix)
    elif bucket_base_prefix:
        prefix = bucket_base_prefix
    else:
        prefix = derive_project_prefix(repo_root, None)
    token, token_source = resolve_token()

    info(f"repo root: {repo_root}")
    info(f"bucket: {bucket_id}")
    info(f"prefix: {prefix}")
    info(f"token source: {token_source}")

    if args.mode == "status":
        info(f"bootstrap: {remote_uri(bucket_id, prefix, 'bootstrap')}")
        info(f"state: {remote_uri(bucket_id, prefix, 'state')}")
        return 0

    if args.mode == "pull":
        pull_bootstrap(args, repo_root, bucket_id, prefix, token)
        pull_state(args, repo_root, bucket_id, prefix, token)
    elif args.mode == "pull-bootstrap":
        pull_bootstrap(args, repo_root, bucket_id, prefix, token)
    elif args.mode == "pull-state":
        pull_state(args, repo_root, bucket_id, prefix, token)
    elif args.mode in {"push", "push-state"}:
        push_state(args, repo_root, bucket_id, prefix, token)
    elif args.mode == "upload-bootstrap":
        upload_bootstrap(args, repo_root, bucket_id, prefix, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
