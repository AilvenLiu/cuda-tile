#!/bin/bash
#
# Given a CUDA Tile IR git repository, an LLVM git repository and an LLVM commit
# hash, this script determines the appropriate CUDA Tile IR open-source version
# to use.
#
# See the "Versioning" and "Keeping Compatibility with LLVM" sections in
# README.md for details on how CUDA Tile IR manages versioning and compatibility
# with LLVM.
#
# Usage:
#   ./get-cuda-tile-version-for-llvm-hash.sh \
#     --cuda-tile-repo /path/to/cuda-tile \
#     --llvm-repo /path/to/llvm-project \
#     --llvm-commit <commit-hash>
#

set -e

# =============================================================================
# Constants
# =============================================================================

# LLVM commit hash from the first release (v13.1.0).
FIRST_COMPATIBLE_LLVM_COMMIT="81b576e66bf223f7afc8a86463226cbf1bd480fd"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# Utilities
# =============================================================================

error() { echo -e "${RED}Error: $1${NC}" >&2; exit 1; }
info()  { echo -e "${GREEN}$1${NC}" >&2; }
warn()  { echo -e "${YELLOW}$1${NC}" >&2; }

# Resolve a commit reference to its full SHA hash.
# Args: $1=repo_path, $2=commit_ref
# Returns: Full commit hash or empty string if not found
resolve_commit() {
    git -C "$1" rev-parse "$2" 2>/dev/null
}

# Check if commit_a is an ancestor of (or equal to) commit_b.
# Args: $1=repo_path, $2=commit_a, $3=commit_b
# Returns: 0 if true, 1 if false
is_ancestor_or_equal() {
    local repo="$1" commit_a="$2" commit_b="$3"
    local full_a full_b

    full_a=$(resolve_commit "$repo" "$commit_a")
    full_b=$(resolve_commit "$repo" "$commit_b")

    [[ "$full_a" == "$full_b" ]] && return 0
    git -C "$repo" merge-base --is-ancestor "$commit_a" "$commit_b" 2>/dev/null
}

# Extract LLVM_BUILD_COMMIT_HASH from cmake/IncludeLLVM.cmake at a given tag.
# Args: $1=repo_path, $2=tag
# Returns: Commit hash or empty string
get_llvm_commit_from_tag() {
    git -C "$1" show "$2:cmake/IncludeLLVM.cmake" 2>/dev/null | \
        sed -n 's/.*set(LLVM_BUILD_COMMIT_HASH \([a-f0-9]*\)).*/\1/p' | head -n1
}

# Validate that the LLVM commit is not older than the first compatible commit.
# Args: $1=llvm_repo, $2=llvm_commit_full
validate_llvm_commit_not_too_old() {
    local llvm_repo="$1" llvm_commit="$2"
    local first_compatible="$FIRST_COMPATIBLE_LLVM_COMMIT"

    # Check if first compatible commit exists in LLVM repo
    if ! resolve_commit "$llvm_repo" "$first_compatible" >/dev/null; then
        warn "Warning: First compatible commit $first_compatible not found in LLVM repo"
        return 0
    fi

    # Error if input commit is older than first compatible
    if is_ancestor_or_equal "$llvm_repo" "$llvm_commit" "$first_compatible" && \
       [[ "$llvm_commit" != "$(resolve_commit "$llvm_repo" "$first_compatible")" ]]; then
        error "LLVM commit is older than the first compatible commit ($first_compatible). No CUDA Tile IR version is compatible."
    fi
}

# Find the appropriate CUDA Tile IR version for a given LLVM commit.
# Each CUDA Tile IR version's cmake/IncludeLLVM.cmake contains the LLVM commit it was
# built/tested with. We find the latest version whose LLVM commit is an ancestor of
# (or equal to) the input LLVM commit.
# Args: $1=cuda_tile_repo, $2=llvm_repo, $3=llvm_commit_full, $4=tags, $5=base_version
# Outputs: Sets SELECTED_VERSION global variable
find_compatible_version() {
    local cuda_tile_repo="$1" llvm_repo="$2" llvm_commit="$3" tags="$4" base_version="$5"

    SELECTED_VERSION="$base_version"

    info "Scanning CUDA Tile IR tags..."

    for tag in $tags; do
        local tag_llvm_commit
        tag_llvm_commit=$(get_llvm_commit_from_tag "$cuda_tile_repo" "$tag")
        [[ -z "$tag_llvm_commit" ]] && continue

        # Verify commit exists in LLVM repo
        if ! resolve_commit "$llvm_repo" "$tag_llvm_commit" >/dev/null; then
            warn "  Warning: LLVM commit $tag_llvm_commit not found in LLVM repo (tag: $tag)"
            continue
        fi

        info "  $tag -> LLVM commit $tag_llvm_commit"

        # Update version if this tag's LLVM commit is at or before our target
        if is_ancestor_or_equal "$llvm_repo" "$tag_llvm_commit" "$llvm_commit"; then
            SELECTED_VERSION="$tag"
            info "    -> Compatible"
        fi
    done
}

# =============================================================================
# Argument parsing
# =============================================================================

usage() {
    cat <<EOF
Usage: $0 --cuda-tile-repo <path> --llvm-repo <path> --llvm-commit <hash>

Arguments:
  --cuda-tile-repo    Path to the CUDA Tile IR git repository
  --llvm-repo         Path to the LLVM git repository
  --llvm-commit       LLVM commit hash to check compatibility for

Note: The LLVM commit must be a stable commit on the main branch.
      Commits from feature branches or forks may not be compatible.

Example:
  $0 --cuda-tile-repo ./cuda-tile --llvm-repo ./llvm-project --llvm-commit abc123
EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cuda-tile-repo) CUDA_TILE_REPO="$2"; shift 2 ;;
            --llvm-repo)      LLVM_REPO="$2"; shift 2 ;;
            --llvm-commit)    LLVM_COMMIT="$2"; shift 2 ;;
            -h|--help)        usage ;;
            *)                error "Unknown argument: $1" ;;
        esac
    done

    [[ -z "$CUDA_TILE_REPO" ]] && error "Missing --cuda-tile-repo argument"
    [[ -z "$LLVM_REPO" ]]      && error "Missing --llvm-repo argument"
    [[ -z "$LLVM_COMMIT" ]]    && error "Missing --llvm-commit argument"

    [[ -d "$CUDA_TILE_REPO/.git" ]] || error "Not a git repository: $CUDA_TILE_REPO"
    [[ -d "$LLVM_REPO/.git" ]]      || error "Not a git repository: $LLVM_REPO"
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"

    # Resolve and validate LLVM commit
    LLVM_COMMIT_FULL=$(resolve_commit "$LLVM_REPO" "$LLVM_COMMIT") || \
        error "LLVM commit not found: '$LLVM_COMMIT'"
    info "Resolving CUDA Tile IR version for LLVM commit: $LLVM_COMMIT_FULL"

    # Get sorted version tags
    local tags base_version
    tags=$(git -C "$CUDA_TILE_REPO" tag -l 'v*' | sort -V)
    [[ -z "$tags" ]] && error "No version tags found in CUDA Tile repository"

    base_version=$(echo "$tags" | head -n1)
    info "Base CUDA Tile IR version: $base_version"

    # Validate LLVM commit is not too old
    validate_llvm_commit_not_too_old "$LLVM_REPO" "$LLVM_COMMIT_FULL"

    # Find compatible version
    find_compatible_version "$CUDA_TILE_REPO" "$LLVM_REPO" "$LLVM_COMMIT_FULL" "$tags" "$base_version"

    # Print result
    info ""
    info "============================================"
    info "Recommended CUDA Tile IR version: $SELECTED_VERSION"
    info "============================================"

    # Output version to stdout (for scripting)
    echo "$SELECTED_VERSION"
}

main "$@"
