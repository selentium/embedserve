from __future__ import annotations

import re
import sys
from pathlib import Path

_REQUIREMENT_NAME_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_.,-]+\])?==")
_KEEP_NAMES = {"torch"}


def _requirement_name(line: str) -> str | None:
    match = _REQUIREMENT_NAME_PATTERN.match(line)
    if match is None:
        return None
    return match.group(1).lower().replace("_", "-")


def _split_prefix_and_blocks(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    prefix_lines: list[str] = []
    blocks: list[list[str]] = []
    current_block: list[str] = []
    saw_requirement = False

    for line in lines:
        if not saw_requirement and _requirement_name(line) is None:
            prefix_lines.append(line)
            continue

        saw_requirement = True
        if not current_block:
            current_block = [line]
            continue

        if line.startswith((" ", "\t")):
            current_block.append(line)
            continue

        blocks.append(current_block)
        current_block = [line]

    if current_block:
        blocks.append(current_block)

    return prefix_lines, blocks


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: python scripts/filter_torch_overlay_requirements.py INPUT OUTPUT BASE_REQUIREMENTS_PATH",
            file=sys.stderr,
        )
        return 1

    input_path = Path(argv[1])
    output_path = Path(argv[2])
    base_requirements_path = argv[3]
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    prefix_lines, blocks = _split_prefix_and_blocks(lines)

    output_lines = list(prefix_lines)
    if output_lines and output_lines[-1].strip():
        output_lines.append("\n")
    output_lines.append(f"-r {base_requirements_path}\n")

    for block in blocks:
        if _requirement_name(block[0]) in _KEEP_NAMES:
            output_lines.extend(block)

    output_path.write_text("".join(output_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
