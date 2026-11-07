"""Scan tracked source and optional Git history for credential material."""

import argparse
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "service token": re.compile(r"(?:hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,})"),
    "AWS access identifier": re.compile(r"AKIA[A-Z0-9]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "literal credential": re.compile(r"(?:password|api[_-]?key|access_token|client_secret)\s*=\s*['\"][A-Za-z0-9_./+=-]{12,}['\"]", re.I),
}


def matches(text):
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    findings = []
    for name in filter(None, names):
        path = root / name
        if path.is_file() and path.name != Path(__file__).name:
            text = path.read_bytes().decode("utf-8", errors="ignore")
            for category in matches(text):
                findings.append(f"{name}: {category}")
    if args.history:
        objects = subprocess.check_output(["git", "-C", str(root), "rev-list", "--objects", "--all"]).decode().splitlines()
        for entry in objects:
            parts = entry.split(" ", 1)
            if len(parts) != 2 or parts[1].endswith("privacy_scan.py"):
                continue
            obj, name = parts
            kind = subprocess.check_output(["git", "-C", str(root), "cat-file", "-t", obj]).strip()
            if kind != b"blob":
                continue
            raw = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", obj])
            for category in matches(raw.decode("utf-8", errors="ignore")):
                findings.append(f"Historical source {name}: {category}")
    if findings:
        for finding in sorted(set(findings)):
            print(finding)
        raise SystemExit(1)
    print("Credential scan passed for tracked source" + (" and Git history." if args.history else "."))


if __name__ == "__main__":
    main()
