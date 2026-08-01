#!/usr/bin/env python3
"""Validate public-repository hygiene and the Kaggle notebook contract."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle-t4x2-text-vision.ipynb"
LICENSE = ROOT / "LICENSE"
AUTHOR = "Đăng Khoa <i.am@dangkhoa.dev>"

CANONICAL_NAME = "KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision"
CANONICAL_URL = (
    "https://github.com/dangkhoa2016/"
    "KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision"
)
OBSOLETE_NAME = "KerasHub-TranslateGemma-4B-IT-Kaggle-T4x2-Text-Vision"
OBSOLETE_URL = (
    "https://github.com/dangkhoa2016/"
    "KerasHub-TranslateGemma-4B-IT-Kaggle-T4x2-Text-Vision"
)

# Files where the canonical repository name/URL is expected to be used.
CANONICAL_FILES = (
    ROOT / "README.md",
    ROOT / "README.vi.md",
    ROOT / "NOTICE.md",
    ROOT / "NOTICE.vi.md",
    NOTEBOOK,
)

FORBIDDEN_TRACKED_PATHS = {
    ".env",
    "data/api_key.txt",
    "data/restart_secret.txt",
    "data/tunnel_url.txt",
    "data/environment.json",
    "bin/cloudflared",
}
FORBIDDEN_NOTEBOOK_PHRASES = (
    "authorized_keys",
    "ngrok_authtoken",
    "ssh_host_",
)
VIETNAMESE_MARKERS = re.compile(
    r"\b(?:hãy|không|chuẩn bị|kiểm tra|dừng|hoàn tất|giải nén|đóng gói)\b",
    re.IGNORECASE,
)
# Lines carrying one of these markers are treated as intentional historical
# documentation of the old repository name and are not flagged.
HISTORICAL_MARKERS = (
    "historically",
    "formerly",
    "previous name",
    "old name",
    "previously known",
    "renamed",
    "was known as",
    "tên cũ",
    "tên trước đây",
    "trước đây gọi",
    "được đổi tên",
)


def validate_license() -> None:
    license_text = LICENSE.read_text(encoding="utf-8")
    if not license_text.startswith("MIT License\n"):
        raise SystemExit("LICENSE must contain the canonical MIT License text.")
    if AUTHOR not in license_text:
        raise SystemExit(
            "LICENSE does not contain the expected author/copyright holder."
        )


def _tracked_paths() -> set[Path]:
    """Return repository files tracked by git (ignored scratch is excluded)."""
    output = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    )
    return {ROOT / rel for rel in output.splitlines()}


def validate_bilingual_markdown() -> None:
    tracked = _tracked_paths()
    for english in sorted(ROOT.rglob("*.md")):
        if english not in tracked:
            continue
        if english.name.endswith(".vi.md"):
            continue
        vietnamese = english.with_name(f"{english.stem}.vi.md")
        if not vietnamese.is_file():
            raise SystemExit(
                f"Missing Vietnamese Markdown counterpart: "
                f"{vietnamese.relative_to(ROOT)}"
            )

        en_link = f"> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt]({vietnamese.name})"
        if en_link not in english.read_text(encoding="utf-8"):
            raise SystemExit(
                f"Missing English language switch: {english.relative_to(ROOT)}"
            )

        vi_link = f"> 🌐 Language / Ngôn ngữ: [English]({english.name}) | **Tiếng Việt**"
        if vi_link not in vietnamese.read_text(encoding="utf-8"):
            raise SystemExit(
                f"Missing Vietnamese language switch: {vietnamese.relative_to(ROOT)}"
            )


def _notebook_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells") or []
    if not cells or cells[0].get("cell_type") != "code":
        raise SystemExit(
            "The first notebook cell must be the repository bootstrap code cell."
        )
    return cells


def _collect_violations(lines, rel, obsolete_values) -> list:
    """Return readable (location, found, replacement) tuples for a file."""
    violations = []
    for lineno, line in enumerate(lines, start=1):
        if any(marker.lower() in line.lower() for marker in HISTORICAL_MARKERS):
            continue
        for found, canonical in obsolete_values:
            if found in line:
                violations.append((f"{rel}:{lineno}", found, canonical))
                break
    return violations


def validate_canonical_identity() -> None:
    """Reject accidental use of the obsolete repository name/URL where the
    canonical values are expected, while allowing intentional historical notes."""
    canonical_by_file = {
        ROOT / "README.md": (CANONICAL_NAME, CANONICAL_URL),
        ROOT / "README.vi.md": (CANONICAL_NAME, CANONICAL_URL),
        ROOT / "NOTICE.md": (CANONICAL_NAME,),
        ROOT / "NOTICE.vi.md": (CANONICAL_NAME,),
    }
    for path, expected in canonical_by_file.items():
        text = path.read_text(encoding="utf-8")
        for value in expected:
            if value not in text:
                raise SystemExit(
                    f"{path.relative_to(ROOT)} does not contain the canonical "
                    f"repository identity: {value}"
                )

    obsolete_values = ((OBSOLETE_URL, CANONICAL_URL), (OBSOLETE_NAME, CANONICAL_NAME))
    violations: list = []
    for path in CANONICAL_FILES:
        if path == NOTEBOOK:
            lines = []
            for cell in _notebook_cells():
                lines.extend("".join(cell.get("source") or []).splitlines())
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
        violations.extend(
            _collect_violations(lines, path.relative_to(ROOT), obsolete_values)
        )

    if violations:
        for location, found, canonical in violations:
            print(
                f"Obsolete repository reference found in {location}:\n"
                f"  found:      {found}\n"
                f"  replace with: {canonical}"
            )
        raise SystemExit(
            "Obsolete repository name/URL detected; update to the canonical value."
        )


def main() -> None:
    validate_license()
    validate_bilingual_markdown()
    validate_canonical_identity()

    for rel in FORBIDDEN_TRACKED_PATHS:
        if (ROOT / rel).exists():
            raise SystemExit(f"Forbidden runtime or secret path is present: {rel}")

    cells = _notebook_cells()
    first_source = "".join(cells[0].get("source") or [])
    if "git clone" not in first_source or CANONICAL_URL not in first_source:
        raise SystemExit(
            "The first notebook cell does not clone the canonical repository URL."
        )
    if CANONICAL_NAME not in first_source:
        raise SystemExit(
            "The first notebook cell does not use the canonical repository "
            "name for the working directory."
        )

    source_text = "\n".join("".join(cell.get("source") or []) for cell in cells)
    lowered = source_text.lower()
    for phrase in FORBIDDEN_NOTEBOOK_PHRASES:
        if phrase.lower() in lowered:
            raise SystemExit(f"Forbidden notebook phrase found: {phrase}")
    if VIETNAMESE_MARKERS.search(source_text):
        raise SystemExit(
            "Vietnamese instructional text was found in the public notebook."
        )

    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            raise SystemExit(f"Notebook code cell {index} contains saved output.")

    print("Public repository validation passed.")


if __name__ == "__main__":
    main()
