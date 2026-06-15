# Releasing fast-nnt

`fast-nnt` ships as three packages that share a single version number:

| Package    | Registry  | Built from                     |
|------------|-----------|--------------------------------|
| `fast-nnt` | crates.io | repo root (`Cargo.toml`)       |
| `fastnntpy`| PyPI      | `fastnnt-py/` (maturin / PyO3) |
| `fastnntr` | r-universe| `fastnntr/` (extendr)          |

The two binding crates depend on the **published** core via
`fast-nnt = ">=<VERSION>"` (their `[patch.crates-io]` blocks are commented out).
This means the bindings cannot resolve a new version until the core crate is
live on crates.io. The release flow below is ordered around that constraint.

## Version synchronisation

`release.sh` keeps these five files in lockstep — do not edit them by hand:

- `Cargo.toml` (core)
- `fastnnt-py/Cargo.toml`, `fastnnt-py/pyproject.toml`
- `fastnntr/src/rust/Cargo.toml`, `fastnntr/DESCRIPTION`

It also bumps the `fast-nnt = ">=<VERSION>"` dependency in both binding crates.

## Release flow

### 1. Dry run (validates everything, publishes nothing)

```bash
./release.sh --dry-run 0.3.0
```

This bumps versions, runs `cargo test`, `cargo publish --dry-run`, then **builds
and smoke-tests both bindings against the local core** by temporarily
uncommenting their `[patch.crates-io] → path` block. The patch is reverted on
exit (even on failure), so the working tree is left with only the version bumps.

> The R smoke-test needs a toolchain that can compile the extendr binding
> (Cargo + a working R dev environment). If `Rscript` is missing the R step is
> skipped with a warning; if it is present but cannot compile, run the dry run
> from an environment where `R CMD INSTALL fastnntr` succeeds.

### 2. Real release (publishes core, then bindings, then tags)

```bash
./release.sh 0.3.0
```

Order of operations:

1. Bump + validate versions, `cargo test`, `cargo publish --dry-run`.
2. `cargo publish` the core crate to crates.io.
3. Poll `cargo search fast-nnt` until the new version is visible.
4. `cargo update` + build + smoke-test the Python and R bindings **against the
   published crate** (no path patch).
5. `git add -A && git commit -m "release: vX.Y.Z"`, `git tag vX.Y.Z`, push both.

The tag push triggers the GitHub Actions workflow that builds and uploads the
PyPI wheels.

### Recovering from crates.io index lag

If step 3 times out, the bindings build in step 4 may fail because the new
version is not yet in the index. Once `cargo search fast-nnt` shows the new
version, finish the bindings manually:

```bash
(cd fastnnt-py && cargo update && maturin develop)
(cd fastnntr/src/rust && cargo update) && R CMD INSTALL fastnntr
```

then commit, tag, and push as in step 5.

## R distribution: r-universe

`fastnntr` is distributed through [r-universe](https://r-universe.dev), which
rebuilds the package from this repository's `main` branch whenever it changes.
r-universe builds with network access, so the package compiles the Rust core
live from crates.io — **no `vendor.tar.xz` is required** (the `Makevars` only
uses one if it happens to be present).

### One-time setup

1. Create a public GitHub repo named `rhysnewell.r-universe.dev`.
2. Add a single `packages.json` at its root:

   ```json
   [
     {
       "package": "fastnntr",
       "url": "https://github.com/rhysnewell/fast-nnt",
       "subdir": "fastnntr"
     }
   ]
   ```

3. r-universe picks it up automatically and builds `fastnntr` (fetching the
   published `fast-nnt` core). Users then install with:

   ```r
   install.packages("fastnntr", repos = "https://rhysnewell.r-universe.dev")
   ```

### Ongoing releases

After a normal `./release.sh X.Y.Z` lands the version bump on `main`, r-universe
detects the change and rebuilds `fastnntr` automatically — no extra step is
required beyond making sure the core crate is published first (which the release
flow guarantees).
