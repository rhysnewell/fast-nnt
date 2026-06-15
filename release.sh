#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VERSION_FILES=(
  "Cargo.toml"
  "fastnnt-py/Cargo.toml"
  "fastnnt-py/pyproject.toml"
  "fastnntr/src/rust/Cargo.toml"
  "fastnntr/DESCRIPTION"
)

DEP_FILES=(
  "fastnnt-py/Cargo.toml"
  "fastnntr/src/rust/Cargo.toml"
)

# ── Helpers ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}==> $*${NC}"; }
warn()  { echo -e "${YELLOW}WARNING: $*${NC}"; }
error() { echo -e "${RED}ERROR: $*${NC}" >&2; }
fatal() { error "$@"; exit 1; }

# ── Local path-patch management (dry-run only) ───────────────────────────────
# The binding crates ship with a commented-out [patch.crates-io] block that
# points fast-nnt at the local core. For a dry-run we temporarily uncomment it
# so the bindings build against the unpublished local core, then restore the
# original files (even on failure, via the EXIT trap).
PATCH_FILES=(
  "fastnnt-py/Cargo.toml"
  "fastnntr/src/rust/Cargo.toml"
)
PATCH_ENABLED=false

enable_local_patches() {
  local rel_path file lock
  for rel_path in "${PATCH_FILES[@]}"; do
    file="${REPO_ROOT}/${rel_path}"
    lock="${file%/Cargo.toml}/Cargo.lock"
    cp "$file" "${file}.relbak"
    [[ -f "$lock" ]] && cp "$lock" "${lock}.relbak"
    # Uncomment "# [patch.crates-io]" and the "# fast-nnt = { path = ... }" line.
    sed -i '' \
      -e 's|^# \[patch.crates-io\]|[patch.crates-io]|' \
      -e 's|^# \(fast-nnt = { path = .* }\)|\1|' \
      "$file"
    if ! grep -q '^\[patch.crates-io\]' "$file"; then
      fatal "Failed to enable local path patch in $rel_path (expected a commented [patch.crates-io] block)"
    fi
    echo "  $rel_path: enabled local path patch"
  done
  PATCH_ENABLED=true
}

restore_local_patches() {
  [[ "$PATCH_ENABLED" == true ]] || return 0
  local rel_path file lock
  for rel_path in "${PATCH_FILES[@]}"; do
    file="${REPO_ROOT}/${rel_path}"
    lock="${file%/Cargo.toml}/Cargo.lock"
    [[ -f "${file}.relbak" ]] && mv "${file}.relbak" "$file"
    [[ -f "${lock}.relbak" ]] && mv "${lock}.relbak" "$lock"
  done
  PATCH_ENABLED=false
}

# Build + smoke-test both bindings. cargo update re-locks each binding against
# whichever fast-nnt the manifest now resolves to (local patch or published).
build_and_test_bindings() {
  info "Building Python bindings..."
  (
    cd "${REPO_ROOT}/fastnnt-py"
    cargo update 2>&1
    maturin develop 2>&1
  )
  info "Smoke-testing Python bindings..."
  python3 -c "import fastnntpy; print('  fastnntpy import: OK ✓')"

  if [[ "$HAS_R" == true ]]; then
    info "Building R bindings..."
    (
      cd "${REPO_ROOT}/fastnntr/src/rust"
      cargo update 2>&1
    )
    R CMD INSTALL "${REPO_ROOT}/fastnntr" 2>&1
    info "Smoke-testing R bindings..."
    Rscript -e "library(fastnntr); cat('  fastnntr import: OK ✓\n')"
  else
    warn "Skipping R bindings (Rscript not available)"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] <VERSION>

Release fast-nnt across Cargo, PyPI, and R.

Arguments:
  VERSION     Target version (e.g. 0.3.0)

Options:
  --dry-run   Run all checks and local builds, but do not publish or push

Steps performed:
  1. Validate prerequisites (cargo, maturin, Rscript, git)
  2. Bump version in all 5 package files + fast-nnt dep to >=VERSION
  3. Validate version consistency across all packages
  4. Run cargo test
  5. cargo publish --dry-run (core)
  6. Build + smoke-test the bindings against the new core:
       --dry-run : via a temporary [patch.crates-io] -> local core
       release   : publish core first, wait for crates.io, then build
                   the bindings against the published crate
  7. [release only] git commit + tag v<VERSION> + push (triggers PyPI CI)

Ordering rationale: the binding crates depend on fast-nnt = ">=VERSION"
from crates.io, so they cannot resolve the new version until core is
published. A real release therefore publishes core *before* building the
bindings; a dry-run substitutes a local path patch so it can validate the
bindings without publishing anything.
EOF
  exit 1
}

# Extract version from a file based on its format
extract_version() {
  local file="$1"
  case "$file" in
    *DESCRIPTION)
      grep '^Version:' "$file" | sed 's/Version: *//'
      ;;
    *pyproject.toml)
      grep '^version' "$file" | head -1 | sed 's/version *= *"\(.*\)"/\1/'
      ;;
    *)
      # Cargo.toml — take the first version line in [package] section
      # Handle both single and double quotes
      sed -n '/^\[package\]/,/^\[/{
        s/^version *= *'\''\([^'\'']*\)'\''/\1/p
        s/^version *= *"\([^"]*\)"/\1/p
      }' "$file" | head -1
      ;;
  esac
}

# Bump version in a file
bump_version() {
  local file="$1"
  local old_ver="$2"
  local new_ver="$3"

  case "$file" in
    *DESCRIPTION)
      sed -i '' "s/^Version: *${old_ver}/Version: ${new_ver}/" "$file"
      ;;
    *pyproject.toml)
      sed -i '' "s/^version = \"${old_ver}\"/version = \"${new_ver}\"/" "$file"
      ;;
    *)
      # Cargo.toml — replace version in [package] section, preserving quote style
      sed -i '' "/^\[package\]/,/^\[/ {
        s/^version = '${old_ver}'/version = '${new_ver}'/
        s/^version = \"${old_ver}\"/version = \"${new_ver}\"/
      }" "$file"
      ;;
  esac
}

# Check that the fast-nnt dependency version is compatible
check_dep_compatibility() {
  local file="$1"
  local target_ver="$2"

  local dep_spec
  dep_spec=$(grep 'fast-nnt' "$file" | grep -v '^\[' | grep -v '^#' | grep -v 'path' | head -1 || true)
  if [[ -z "$dep_spec" ]]; then
    return 0  # no dep line found (might be patched)
  fi

  # Extract version requirement (e.g. ">=0.2.4" or "0.2.5")
  local req
  req=$(echo "$dep_spec" | sed 's/.*["\x27]\([^"'\'']*\)["\x27].*/\1/')

  # Simple check: if it starts with >= extract the min version
  if [[ "$req" == ">="* ]]; then
    local min_ver="${req#>=}"
    # Compare: target must be >= min
    if [[ "$(printf '%s\n%s' "$min_ver" "$target_ver" | sort -V | head -1)" != "$min_ver" ]]; then
      fatal "Dependency in $file requires fast-nnt $req, but target version is $target_ver"
    fi
  fi
}

# ── Parse arguments ──────────────────────────────────────────────────────────
DRY_RUN=false
VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    -*) fatal "Unknown option: $1" ;;
    *)
      [[ -n "$VERSION" ]] && fatal "Unexpected argument: $1"
      VERSION="$1"; shift
      ;;
  esac
done

[[ -z "$VERSION" ]] && usage

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  fatal "Version must be in semver format (e.g. 1.2.3), got: $VERSION"
fi

cd "$REPO_ROOT"

# ── Step 1: Validate prerequisites ──────────────────────────────────────────
info "Checking prerequisites..."

for cmd in cargo maturin git; do
  command -v "$cmd" &>/dev/null || fatal "'$cmd' not found in PATH"
done

# R is optional but warn if missing
if ! command -v Rscript &>/dev/null; then
  warn "Rscript not found — R bindings build will be skipped"
  HAS_R=false
else
  HAS_R=true
fi

# Check git state
if [[ -n "$(git status --porcelain)" ]]; then
  warn "Working directory has uncommitted changes"
  if [[ "$DRY_RUN" == false ]]; then
    echo "  Uncommitted changes will be included in the release commit."
    echo "  Press Ctrl-C to abort, or Enter to continue..."
    read -r
  fi
fi

# Check we're on main (warn only)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
  warn "Not on main branch (on '$BRANCH'). Releases are typically from main."
fi

# Check tag doesn't already exist
if git tag -l "v${VERSION}" | grep -q "v${VERSION}"; then
  fatal "Tag v${VERSION} already exists"
fi

# ── Step 2: Bump versions ───────────────────────────────────────────────────
info "Bumping versions to ${VERSION}..."

for rel_path in "${VERSION_FILES[@]}"; do
  file="${REPO_ROOT}/${rel_path}"
  old_ver=$(extract_version "$file")
  if [[ "$old_ver" == "$VERSION" ]]; then
    echo "  $rel_path: already at $VERSION"
  else
    echo "  $rel_path: $old_ver -> $VERSION"
    bump_version "$file" "$old_ver" "$VERSION"
  fi
done

# ── Step 2b: Bump fast-nnt dependency in FFI binding crates ─────────────────
info "Bumping fast-nnt dependency to >=${VERSION}..."

for rel_path in "${DEP_FILES[@]}"; do
  file="${REPO_ROOT}/${rel_path}"
  # Replace fast-nnt = ">=X.Y.Z" with fast-nnt = ">=VERSION"
  if grep -q 'fast-nnt = ">=' "$file"; then
    sed -i '' "s/fast-nnt = \">=[^\"]*\"/fast-nnt = \">=${VERSION}\"/" "$file"
    echo "  $rel_path: fast-nnt dependency -> >=${VERSION}"
  fi
done

# ── Step 3: Validate version consistency ────────────────────────────────────
info "Validating version consistency..."

MISMATCH=false
for rel_path in "${VERSION_FILES[@]}"; do
  file="${REPO_ROOT}/${rel_path}"
  actual=$(extract_version "$file")
  if [[ "$actual" != "$VERSION" ]]; then
    error "  $rel_path: expected $VERSION, got '$actual'"
    MISMATCH=true
  else
    echo "  $rel_path: $actual ✓"
  fi
done

if [[ "$MISMATCH" == true ]]; then
  fatal "Version mismatch detected. Fix the files above and retry."
fi

# Check dependency compatibility
info "Checking fast-nnt dependency compatibility..."
for rel_path in "${DEP_FILES[@]}"; do
  file="${REPO_ROOT}/${rel_path}"
  check_dep_compatibility "$file" "$VERSION"
  echo "  $rel_path: dependency compatible ✓"
done

# ── Step 4: Run tests ──────────────────────────────────────────────────────
info "Running cargo test..."
cargo test 2>&1

# ── Step 5: Dry-run publish ─────────────────────────────────────────────────
info "Running cargo publish --dry-run..."
cargo publish --dry-run --allow-dirty 2>&1

# ── Step 6: Build + smoke-test bindings against the new core ─────────────────
if [[ "$DRY_RUN" == true ]]; then
  # No 0.x.y on crates.io yet — patch the bindings to the local core so we can
  # still validate that they build and import. Restored on EXIT.
  info "Enabling temporary local path patch for binding builds..."
  trap restore_local_patches EXIT
  enable_local_patches
  build_and_test_bindings
  restore_local_patches
  trap - EXIT

  echo ""
  echo -e "${BOLD}═══ DRY RUN COMPLETE ═══${NC}"
  echo ""
  echo "All checks passed (bindings validated against local core)."
  echo "To perform the actual release, run:"
  echo ""
  echo "  ./release.sh ${VERSION}"
  echo ""
  echo "This will:"
  echo "  1. cargo publish                         (publish fast-nnt ${VERSION})"
  echo "  2. build + smoke-test bindings against published ${VERSION}"
  echo "  3. git commit + tag v${VERSION} + push   (triggers PyPI CI)"
  echo ""
  exit 0
fi

# ── Actual release: publish core BEFORE building the bindings ────────────────
info "Publishing fast-nnt ${VERSION} to crates.io..."
cargo publish --allow-dirty 2>&1

info "Waiting for crate to appear on crates.io..."
PUBLISHED=false
for i in $(seq 1 24); do
  if cargo search fast-nnt 2>/dev/null | grep -q "\"${VERSION}\""; then
    echo "  fast-nnt ${VERSION} found on crates.io ✓"
    PUBLISHED=true
    break
  fi
  sleep 5
done
if [[ "$PUBLISHED" == false ]]; then
  warn "Crate not visible after 120s; the crates.io index may lag."
  warn "Binding builds below need the published crate and may fail until it"
  warn "propagates. Re-run 'cd fastnnt-py && cargo update && maturin develop'"
  warn "(and the R equivalent) once 'cargo search fast-nnt' shows ${VERSION}."
fi

# Now that core is published, the bindings can resolve fast-nnt = ">=${VERSION}".
build_and_test_bindings

info "Committing and tagging..."
git add -A
git diff --cached --quiet || git commit -m "release: v${VERSION}"
git tag "v${VERSION}"

info "Pushing to origin..."
git push origin "${BRANCH}"
git push origin "v${VERSION}"

echo ""
echo -e "${BOLD}═══ RELEASE v${VERSION} COMPLETE ═══${NC}"
echo ""
echo "  Cargo:  https://crates.io/crates/fast-nnt/${VERSION}"
echo "  PyPI:   GitHub Actions triggered by tag push"
echo "          Check: https://github.com/rhysnewell/fast-nnt/actions"
echo "  R:      Already installed locally"
echo ""
echo "Monitor PyPI wheel builds at the GitHub Actions link above."
