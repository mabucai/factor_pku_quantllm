"""Check staged Git content (or reachable history) without printing secret values.

This is a heuristic submission guard, not proof that arbitrary files are safe.
Review screenshots, documents and custom exports manually before publishing.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import PurePosixPath


SENSITIVE_KEYS = {
    "username", "userid", "investorid", "accountid", "account", "password",
    "passwd", "pwd", "authcode", "apikey", "secretkey", "accesskey",
    "accesskeyid", "secretaccesskey", "awsaccesskeyid", "awssecretaccesskey",
    "token", "accesstoken", "refreshtoken", "clientsecret", "appid",
    "用户名", "账号", "账户", "密码", "授权编码", "密钥", "产品名称",
}
TOKENS = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b|"
    r"\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"https?://[^\s/:]+:[^\s/@]+@"
)
ASSIGNMENT = re.compile(
    r'''(?i)["']?(用户名|账号|密码|授权编码|产品名称|(?:ctp_)?(?:user_?id|password|auth_?code)|'''
    r'''api_?key|access_?token|client_?secret|investor_?id)["']?\s*[:=]\s*["']([^"'\r\n]+)["']'''
)
SNAPSHOT = re.compile(
    r"(?is)(?:balance|余额)\s*[:=]\s*[-+]?\d.*?"
    r"(?:frozen|available|冻结|可用)\s*[:=]\s*[-+]?\d"
)


def sensitive(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", key.lower())
    return normalized in SENSITIVE_KEYS or any(
        normalized.endswith(suffix)
        for suffix in ("password", "authcode", "apikey", "secretkey", "accesstoken")
    )


def allowed(value: object) -> bool:
    return value is None or value == "" or (
        isinstance(value, str) and re.fullmatch(r"\$\{[A-Z][A-Z0-9_]*\}", value) is not None
    )


def forbidden_path(name: str) -> bool:
    path = PurePosixPath(name.lower())
    base = path.name
    if base.endswith(".example.json") or base == ".env.example":
        return False
    return (
        base in {"config.json", "local_config.json", ".env"}
        or base.startswith(".env.")
        or ".local." in base
        or any(part in {"logs", "log", "private", "snapshots", "exports", "secrets", "credentials"} for part in path.parts)
        or (".vntrader" in path.parts and base != ".gitkeep")
        or ("config" in path.parts and path.suffix in {".json", ".yaml", ".yml", ".toml", ".ini"})
        or base.startswith(("account_snapshot", "fund_snapshot", "settlement"))
        or path.suffix in {".pem", ".key", ".p12", ".pfx", ".log", ".db", ".sqlite", ".sqlite3", ".con", ".pkl", ".pickle", ".zip", ".7z", ".tar", ".gz"}
    )


def inspect_content(name: str, data: bytes) -> list[str]:
    findings = set()
    if forbidden_path(name):
        findings.add("private/runtime file is tracked")
    if b"\0" in data:
        return sorted(findings)  # Binary artifacts require manual visual review.
    text = data.decode("utf-8-sig", errors="replace")
    if TOKENS.search(text):
        findings.add("possible credential/token")
    if SNAPSHOT.search(text):
        findings.add("possible account balance snapshot")
    if any(not allowed(m.group(2)) for m in ASSIGNMENT.finditer(text)):
        findings.add("non-empty credential assignment")

    def walk(value: object) -> None:
        if isinstance(value, dict):
            numeric = {key.lower() for key, item in value.items() if isinstance(item, (int, float))}
            if numeric.intersection({"balance", "余额"}) and numeric.intersection({"frozen", "available", "冻结", "可用"}):
                findings.add("possible account balance snapshot")
            for key, item in value.items():
                if sensitive(key) and not allowed(item):
                    findings.add("non-empty credential field")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    if name.endswith(".json"):
        try:
            parsed = json.loads(text)
            walk(parsed)
            if name.endswith(".example.json"):
                def empty_template(item: object) -> bool:
                    if isinstance(item, dict):
                        return all(empty_template(v) for v in item.values())
                    if isinstance(item, list):
                        return all(empty_template(v) for v in item)
                    return item is None or item == "" or item is False
                if not empty_template(parsed):
                    findings.add("example must contain only empty values (false is allowed)")
        except json.JSONDecodeError:
            findings.add("invalid JSON; review manually")
    if name.endswith(".py"):
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and isinstance(key.value, str) and sensitive(key.value):
                            if isinstance(value, ast.Constant) and not allowed(value.value):
                                findings.add("hardcoded credential literal")
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if isinstance(node.value, ast.Constant) and not allowed(node.value.value):
                        if any(isinstance(t, ast.Name) and sensitive(t.id) for t in targets):
                            findings.add("hardcoded credential literal")
        except SyntaxError:
            findings.add("invalid Python; review manually")
    return sorted(findings)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Scan unique blobs reachable from all local refs.")
    args = parser.parse_args()
    entries = set()
    if args.history:
        for commit in git("rev-list", "--all").decode().splitlines():
            for row in git("ls-tree", "-rz", commit).split(b"\0"):
                if row:
                    metadata, path = row.split(b"\t", 1)
                    _, kind, oid = metadata.split()
                    if kind == b"blob":
                        entries.add((path.decode("utf-8"), oid.decode()))
    else:
        for row in git("ls-files", "--stage", "-z").split(b"\0"):
            if row:
                metadata, path = row.split(b"\t", 1)
                _, oid, _ = metadata.split()
                entries.add((path.decode("utf-8"), oid.decode()))
    failures = 0
    binary_count = 0
    for name, oid in sorted(entries):
        data = git("cat-file", "blob", oid)
        binary_count += b"\0" in data
        for reason in inspect_content(name, data):
            print(f"FAIL {name} [{oid[:10]}]: {reason}")
            failures += 1
    print(f"Checked {len(entries)} file versions; {failures} findings; {binary_count} binary artifacts need manual review.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
