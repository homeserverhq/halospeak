#!/usr/bin/env python3
"""Audit installed Python package licenses for distribution readiness.

The audit is intentionally dependency-free so it can run inside the CPU Docker
image after pip installation. It reports package name, version, declared
license, and flags a small denylist of GPU-only or research-only components.

Usage:
    python scripts/audit_licenses.py
    python scripts/audit_licenses.py --write THIRD_PARTY_LICENSES.md
    python scripts/audit_licenses.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageMetadata, distributions
from pathlib import Path

# Packages that must not ship in the CPU/commercial image.
DENYLIST = {
    "demucs",
    "dora-search",
    "lameenc",
    "onnxruntime-rocm",
    "openunmix",
    "submitit",
    "treetable",
}

# Packages that ship license text in their files but omit metadata fields.
KNOWN_LICENSES = {
    "apex": "BSD-3-Clause",
    "ffmpeg-python": "Apache-2.0",
    "halospeak": "MIT",
    "primepy": "MIT",
    "pyannote-audio": "MIT",
    "pyannote-core": "MIT",
    "pyannote-database": "MIT",
    "pyannote-metrics": "MIT",
    "pyannote-pipeline": "MIT",
    "pyannoteai-sdk": "MIT",
    "pytorch-metric-learning": "MIT",
    "sherpa-onnx": "Apache-2.0",
    "sherpa-onnx-core": "Apache-2.0",
}

OSI_LICENSE_PATTERN = re.compile(
    r"\b(MIT|BSD|Apache|GPL|LGPL|AGPL|MPL|ISC|PSF|Python Software Foundation|"
    r"Unlicense|CC0|Zlib|BSL|Artistic|EPL|Public Domain)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PackageLicense:
    """License details for one installed package."""

    name: str
    version: str
    license: str
    osi_compatible: bool


def _first_non_empty(*values: object) -> str:
    """Return the first non-empty string value.

    Args:
        *values: Candidate values.

    Returns:
        The first non-empty stripped string, or an empty string.
    """
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _meta_get(metadata: PackageMetadata, key: str) -> str:
    """Return a package metadata value or an empty string.

    Args:
        metadata: Distribution metadata.
        key: Metadata field name.

    Returns:
        Field value, or an empty string when absent.
    """
    try:
        value = metadata[key]
    except KeyError:
        return ""
    return value.strip() if value else ""


def _declared_license(dist: Distribution, name: str) -> str:
    """Return a package's declared license.

    Args:
        dist: Installed distribution.
        name: Normalized package name.

    Returns:
        A license identifier or short description.
    """
    metadata = dist.metadata
    classifiers = [
        classifier
        for classifier in (metadata.get_all("Classifier") or [])
        if classifier.startswith("License")
    ]
    known = KNOWN_LICENSES.get(name.lower().replace("_", "-").replace(".", "-"))
    declared = _first_non_empty(
        known,
        _meta_get(metadata, "License-Expression"),
        _meta_get(metadata, "License"),
        "; ".join(classifiers),
    )
    if not declared or declared.upper() == "UNKNOWN":
        return "UNKNOWN"
    first_line = declared.splitlines()[0].strip()
    return first_line[:160]


def _collect() -> list[PackageLicense]:
    """Collect license details for all installed distributions.

    Returns:
        Sorted package license records.
    """
    packages: list[PackageLicense] = []
    for dist in distributions():
        raw_name = _meta_get(dist.metadata, "Name")
        name = raw_name.strip()
        if not name:
            continue
        declared = _declared_license(dist, name)
        packages.append(
            PackageLicense(
                name=name,
                version=dist.version,
                license=declared,
                osi_compatible=bool(OSI_LICENSE_PATTERN.search(declared)),
            )
        )
    return sorted(packages, key=lambda item: item.name.lower())


def _print_table(packages: list[PackageLicense]) -> None:
    """Print a human-readable license table.

    Args:
        packages: Package license records.
    """
    print(f"{'Package':40} {'Version':16} License")
    print("-" * 100)
    for package in packages:
        print(
            f"{package.name[:40]:40} {package.version[:16]:16} {package.license[:40]}"
        )


def _markdown(packages: list[PackageLicense]) -> str:
    """Render package license records as Markdown.

    Args:
        packages: Package license records.

    Returns:
        Markdown document.
    """
    lines = [
        "# Runtime Python Package Licenses",
        "",
        "Generated from the installed environment. Review this file before "
        "redistributing the image.",
        "",
        "| Package | Version | Declared license | OSI-compatible metadata |",
        "| -- | -- | -- | -- |",
    ]
    for package in packages:
        lines.append(
            f"| `{package.name}` | {package.version} | "
            f"{package.license} | {'yes' if package.osi_compatible else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _check(packages: list[PackageLicense]) -> list[str]:
    """Return distribution-readiness violations.

    Args:
        packages: Package license records.

    Returns:
        Human-readable violation messages.
    """
    violations: list[str] = []
    installed = {package.name.lower() for package in packages}
    for denied in sorted(DENYLIST & installed):
        violations.append(f"Denylisted package is installed: {denied}")
    for package in packages:
        if not package.osi_compatible and package.license == "UNKNOWN":
            violations.append(
                f"Unknown license for package {package.name}=={package.version}"
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run the license audit.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        type=Path,
        metavar="PATH",
        help="Write the Markdown inventory to PATH instead of stdout.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when denylisted or unknown-license packages exist.",
    )
    args = parser.parse_args(argv)

    packages = _collect()
    if args.write:
        args.write.write_text(_markdown(packages), encoding="utf-8")
        print(f"Wrote {args.write}")
    else:
        _print_table(packages)

    if args.check:
        violations = _check(packages)
        if violations:
            for violation in violations:
                print(f"ERROR: {violation}", file=sys.stderr)
            return 1
        print("License audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
