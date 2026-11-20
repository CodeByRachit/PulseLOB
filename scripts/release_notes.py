#!/usr/bin/env python3
"""Render the GitHub release title and notes for a tag from CHANGELOG.md.

    python3 scripts/release_notes.py v0.4.0 --title          # prints the release title
    python3 scripts/release_notes.py v0.4.0 > notes.md       # prints the release notes

The notes are assembled from the version's CHANGELOG section (summary line, Highlights, Added,
Changed, ...) plus a get-started block, documentation links pinned to the tag (only for documents
that exist at that tag), and a compare link. Used by .github/workflows/release.yml.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "CodeByRachit/PulseLOB"
PHASES = 6
DOCS = [
    ("docs/math_spec.md", "Mathematical specification"),
    ("docs/architecture.md", "Architecture and engineering specification"),
    ("docs/benchmarks.md", "Benchmarks and the decisions they support"),
    ("docs/data.md", "Data sources, formats and known gaps"),
    ("CHANGELOG.md", "Changelog"),
]


def section(version: str) -> tuple[str, str, str]:
    """(summary line, body without the summary, previous version or "")."""
    text = (ROOT / "CHANGELOG.md").read_text()
    heads = list(re.finditer(r"^## \[([^\]]+)\][^\n]*$", text, re.M))
    for k, h in enumerate(heads):
        if h.group(1) == version:
            end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
            body = text[h.end():end]
            body = "\n".join(line for line in body.splitlines() if not re.match(r"^\[[^\]]+\]: ", line)).strip()
            summary, _, rest = body.partition("\n\n")
            prev = heads[k + 1].group(1) if k + 1 < len(heads) and heads[k + 1].group(1) != "Unreleased" else ""
            return summary.strip(), rest.strip(), prev
    raise SystemExit(f"version {version} not found in CHANGELOG.md")


def exists_at(tag: str, path: str) -> bool:
    try:
        subprocess.run(["git", "cat-file", "-e", f"{tag}:{path}"], cwd=ROOT, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return (ROOT / path).exists()


# Short release titles; versions not listed use the CHANGELOG summary line.
TITLES = {
    "0.1.0": "Mathematical specification",
    "0.2.0": "Architecture, build system and data schemas",
    "0.3.0": "C++20 engine, verification and benchmarks",
    "0.4.0": "Python bindings, synthetic data and Binance data tools",
}


def title(tag: str) -> str:
    version = tag.lstrip("v")
    summary, _, _ = section(version)
    short = TITLES.get(version) or re.sub(r"^Phase \d+:\s*", "", summary).rstrip(".")
    return f"{tag} — {short[0].upper() + short[1:]}"


def notes(tag: str) -> str:
    version = tag.lstrip("v")
    summary, body, prev = section(version)
    m = re.match(r"Phase (\d+):\s*(.*)", summary)
    phase = int(m.group(1)) if m else None
    out = []
    if phase is not None:
        out.append(f"> **Phase {phase} of {PHASES}** · {m.group(2)}")
    else:
        out.append(f"> {summary}")
    out.append("")
    body = body.replace("### Highlights", "## Highlights", 1)
    first_detail = re.search(r"^### ", body, re.M)
    if first_detail:
        out.append(body[: first_detail.start()].strip())
        out.append("")
        out.append(get_started(tag))
        out.append("")
        out.append("## What's changed")
        out.append("")
        out.append(body[first_detail.start():].strip())
    else:
        out.append(body)
    out.append("")
    links = [f"- [{label}](https://github.com/{REPO}/blob/{tag}/{path})" for path, label in DOCS if exists_at(tag, path)]
    out.append("## Documentation")
    out.append("")
    out.extend(links)
    out.append("")
    if prev:
        out.append(f"**Full changelog**: https://github.com/{REPO}/compare/v{prev}...{tag}")
    else:
        out.append(f"**Full changelog**: https://github.com/{REPO}/commits/{tag}")
    return "\n".join(out).rstrip() + "\n"


def get_started(tag: str) -> str:
    lines = ["## Get started", "", "```bash", f"git clone --branch {tag} https://github.com/{REPO}.git && cd PulseLOB"]
    if exists_at(tag, "include/pulselob/engine/engine.hpp"):
        lines += ["cmake --preset release && cmake --build --preset release && ctest --preset release"]
    if exists_at(tag, "python/pulselob/engine.py"):
        lines += ["pip install .            # Python package: pulselob.Engine, pulselob.synth, pulselob.data"]
    elif exists_at(tag, "pyproject.toml"):
        lines += ["pip install .            # Python package: schema and .plev files"]
    if exists_at(tag, "research/phase1/validate_math_spec.py"):
        lines += ["pip install -r research/requirements.txt && python research/phase1/validate_math_spec.py"]
    lines.append("```")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    tag = argv[0]
    print(title(tag) if "--title" in argv[1:] else notes(tag), end="" if "--title" not in argv[1:] else "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
