#!/usr/bin/env python3
"""Enforce the component dependency rules of docs/architecture.md Section 2.

Every header under include/pulselob/<component>/ may include only standard headers, headers of
its own component, headers of the components listed for it below, and pulselob/version.hpp.
Public headers may not include third-party code. Exit code 1 on any violation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "feed": {"core"},
    "hawkes": {"core"},
    "control": {"core", "hawkes"},
    "engine": {"core", "feed", "hawkes", "control"},
    "synth": {"core", "feed"},  # Phase 4: synthetic data
    "sim": {"core", "feed", "engine"},  # Phase 5
}
THIRD_PARTY = ("pybind11/", "gtest/", "gmock/", "benchmark/", "Eigen/")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "include" / "pulselob"
    errors: list[str] = []
    for header in sorted(root.rglob("*.hpp")):
        rel = header.relative_to(root)
        if len(rel.parts) < 2:
            continue  # top-level headers such as version.hpp
        component = rel.parts[0]
        if component not in ALLOWED:
            errors.append(f"{rel}: component '{component}' is not in the layering table")
            continue
        for target in INCLUDE_RE.findall(header.read_text()):
            if target.startswith(THIRD_PARTY):
                errors.append(f"{rel}: public header includes third-party '{target}'")
            elif target.startswith("pulselob/"):
                parts = target.split("/")
                if len(parts) == 2:
                    continue  # pulselob/version.hpp
                dep = parts[1]
                if dep != component and dep not in ALLOWED[component]:
                    errors.append(f"{rel}: '{component}' may not depend on '{dep}' ({target})")
    for e in errors:
        print(f"layering: {e}", file=sys.stderr)
    if not errors:
        print("layering check passed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
