"""Check distribution member names without extracting or executing the archive.

Usage: python scripts/check_distribution.py dist
This is an accidental-data-inclusion check, not a secret scanner.
"""
from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


PRIVATE_DIRS = {"data", "models", "cache", "runs", "outputs", "reports", "backup",
                ".git", ".vscode", ".idea", "__pycache__"}
PRIVATE_SUFFIXES = {".wav", ".flac", ".mp3", ".npz", ".npy", ".pkl", ".joblib",
                    ".pt", ".pth", ".onnx", ".pyc", ".pyo", ".pem", ".key"}


def forbidden_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return (
        path.is_absolute()
        or ".." in path.parts
        or bool(PRIVATE_DIRS.intersection(path.parts))
        or path.suffix.lower() in PRIVATE_SUFFIXES
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.name == ".DS_Store"
    )


def inspect_archive(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [name for name in archive.namelist() if forbidden_member(name)]
    with tarfile.open(path, "r:gz") as archive:
        return [member.name for member in archive.getmembers()
                if forbidden_member(member.name) or member.issym() or member.islnk()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    archives = sorted(args.directory.glob("*.whl")) + sorted(args.directory.glob("*.tar.gz"))
    if not archives:
        parser.error("no wheel or source distribution found; run python -m build first")
    failed = False
    for archive in archives:
        rejected = inspect_archive(archive)
        print(f"{archive.name}: {'FAIL' if rejected else 'OK'}")
        for name in rejected:
            print(f"  excluded content found: {name}")
        failed |= bool(rejected)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
