"""Replace a sentinel-delimited section of README.md with content read from STDIN.

The target section in README.md must be wrapped in matching sentinels:

    <!-- BEGIN:benchmarks -->
    ...content that will be replaced...
    <!-- END:benchmarks -->

Everything between the sentinels (exclusive) is replaced with STDIN; the sentinels
themselves are preserved so the section can be regenerated repeatedly.

Requirements:
    (standard library only)

Run:
    python scripts/benchmark.py | python scripts/inject_readme_section.py --section benchmarks

Exits non-zero with a clear message if the sentinels are missing or malformed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"


def inject(readme_path: Path, section: str, new_content: str) -> None:
    """Replace the body between BEGIN/END sentinels for ``section`` in ``readme_path``.

    :raises SystemExit: if the file or either sentinel is missing, or they are out of order.
    """
    if not readme_path.exists():
        raise SystemExit(f"error: README not found at {readme_path}")

    begin = f"<!-- BEGIN:{section} -->"
    end = f"<!-- END:{section} -->"
    text = readme_path.read_text(encoding="utf-8")

    begin_idx = text.find(begin)
    end_idx = text.find(end)

    if begin_idx == -1 or end_idx == -1:
        missing = begin if begin_idx == -1 else end
        raise SystemExit(
            f"error: sentinel {missing!r} not found in {readme_path}. "
            f"Add a '{begin}' / '{end}' pair around the section to inject."
        )
    if end_idx < begin_idx:
        raise SystemExit(
            f"error: '{end}' appears before '{begin}' in {readme_path}; sentinels are out of order."
        )

    body = new_content.strip("\n")
    rebuilt = (
        text[: begin_idx + len(begin)]
        + "\n"
        + body
        + "\n"
        + text[end_idx:]
    )
    readme_path.write_text(rebuilt, encoding="utf-8")
    logger.info("Injected %d chars into section %r of %s", len(body), section, readme_path)


def main(argv: list[str] | None = None) -> None:
    """Parse args, read STDIN, and inject into the README."""
    parser = argparse.ArgumentParser(description="Inject STDIN into a README.md section.")
    parser.add_argument(
        "--section",
        required=True,
        help="Section name matching the <!-- BEGIN:name --> / <!-- END:name --> sentinels.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=README_PATH,
        help="Path to the README to modify (default: repo README.md).",
    )
    args = parser.parse_args(argv)

    new_content = sys.stdin.read()
    if not new_content.strip():
        raise SystemExit("error: no content received on STDIN.")

    inject(args.readme, args.section, new_content)


if __name__ == "__main__":
    main()
