"""Fail when common private-path, identifier, credential, or leakage markers are found."""

import argparse
import re
from pathlib import Path


PATTERNS = {
    "absolute user path": re.compile(r"/(?:Users|home|lvm\d?)/[^/\s]+/", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Z]:\\\\Users\\\\[^\\\\\s]+", re.IGNORECASE),
    "credential assignment": re.compile(r"(?:token|password|secret|api[_-]?key)\s*=\s*['\"][^'\"]+", re.IGNORECASE),
    "test-to-train option": re.compile(r"augment_(?:internal_test|external_test)|internal_test_as_val", re.IGNORECASE),
    "long numeric identifier": re.compile(r"(?<![\d.])\d{12,}(?![\d.])"),
}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".yaml", ".yml", ".json", ".toml"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if ".git" in path.parts:
            continue
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((path.relative_to(root), line, name, match.group(0)))
    if findings:
        for path, line, name, value in findings:
            print(f"{path}:{line}: {name}: {value}")
        raise SystemExit(1)
    print("Privacy scan passed.")


if __name__ == "__main__":
    main()
