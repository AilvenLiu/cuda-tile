# RUN: %PYTHON %s %S/../../README.md %t.mlir
# RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t.mlir -o %t.tilebc
# RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.tilebc -o %t.roundtrip.mlir
# RUN: FileCheck %s --input-file=%t.roundtrip.mlir
# RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %t.roundtrip.mlir -o %t.roundtrip.tilebc

# CHECK-LABEL: entry @example_kernel
# CHECK: print_tko "Running example module\0A" -> token
# CHECK: load_ptr_tko weak
# CHECK: print_tko "Data: %f\0A", %{{.*}} : tile<128xf32> -> token
# CHECK: return

"""Extract the example.mlir snippet from README.md for round-trip testing."""

import pathlib
import sys


START_MARKER = "// README-EXAMPLE-START"
END_MARKER = "// README-EXAMPLE-END"


def extract_example(readme: pathlib.Path) -> str:
    lines = readme.read_text(encoding="utf-8").splitlines()

    if lines.count(START_MARKER) != 1 or lines.count(END_MARKER) != 1:
        raise RuntimeError("expected exactly one pair of README example markers")

    start = lines.index(START_MARKER)
    end = lines.index(END_MARKER)
    if start >= end:
        raise RuntimeError("README example markers are out of order")

    snippet = "\n".join(lines[start + 1 : end]) + "\n"
    if "cuda_tile.module" not in snippet:
        raise RuntimeError("the README example is not a CUDA Tile module")
    return snippet


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <README.md> <output.mlir>")
    pathlib.Path(sys.argv[2]).write_text(
        extract_example(pathlib.Path(sys.argv[1])), encoding="utf-8"
    )
