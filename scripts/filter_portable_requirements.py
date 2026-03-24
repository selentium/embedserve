from __future__ import annotations

import re
import sys
from pathlib import Path

_REQUIREMENT_NAME_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_.,-]+\])?==")
_EXCLUDED_PREFIXES = ("cuda-", "nvidia-")
_EXCLUDED_NAMES = {"triton"}
_TORCH_NAME = "torch"


def _should_exclude(requirement_line: str) -> bool:
    match = _REQUIREMENT_NAME_PATTERN.match(requirement_line)
    if match is None:
        return False

    normalized_name = match.group(1).lower().replace("_", "-")
    return normalized_name.startswith(_EXCLUDED_PREFIXES) or normalized_name in _EXCLUDED_NAMES


def _rewrite_requirement_line(requirement_line: str) -> str:
    match = _REQUIREMENT_NAME_PATTERN.match(requirement_line)
    if match is None:
        return requirement_line

    normalized_name = match.group(1).lower().replace("_", "-")
    if normalized_name != _TORCH_NAME:
        return requirement_line

    return f'{requirement_line.rstrip()} ; sys_platform != "linux"\n'


def _filter_blocks(lines: list[str]) -> list[str]:
    filtered_lines: list[str] = []
    pending_block: list[str] = []

    for line in lines:
        if not pending_block:
            if line.startswith((" ", "\t")):
                filtered_lines.append(line)
                continue

            pending_block = [line]
            continue

        if line.startswith((" ", "\t")):
            pending_block.append(line)
            continue

        if not _should_exclude(pending_block[0]):
            pending_block[0] = _rewrite_requirement_line(pending_block[0])
            filtered_lines.extend(pending_block)
        pending_block = [line]

    if pending_block and not _should_exclude(pending_block[0]):
        pending_block[0] = _rewrite_requirement_line(pending_block[0])
        filtered_lines.extend(pending_block)

    return filtered_lines


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python scripts/filter_portable_requirements.py INPUT OUTPUT",
            file=sys.stderr,
        )
        return 1

    input_path = Path(argv[1])
    output_path = Path(argv[2])
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output_path.write_text("".join(_filter_blocks(lines)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
