#!/usr/bin/env bash
#
# Regenerate every encoding-stability golden in this directory.
#
# Usage:
#   regenerate_goldens.sh [path/to/cuda-tile-translate]
#
# If the translate tool isn't on $PATH, pass it as the first argument.
#
# Run this script ONLY when an encoding shift is intentional (for example,
# a deliberate bytecode-format version bump). A regression that shows up
# in this suite usually means either:
#   (a) MLIR's canonical raw-buffer / float / int encoding silently shifted
#       under us (the i1 0xff bug pattern); add a backward-compat shim in
#       BytecodeReader.cpp before regenerating, OR
#   (b) something on our writer side changed unintentionally; revert it.
#
# Do not run this to "make the failure go away" without first understanding
# which encoding moved and why.

set -euo pipefail

TRANSLATE="${1:-cuda-tile-translate}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for src in "${DIR}"/*.mlir; do
  name="$(basename "${src}" .mlir)"
  out="${DIR}/goldens/${name}.golden.tileirbc"
  echo "Regenerating ${out}"
  "${TRANSLATE}" -mlir-to-cudatilebc -no-implicit-module "${src}" -o "${out}"
done
