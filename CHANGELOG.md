# Changelog

Versions follow [semantic versioning](https://semver.org/). While the major
version is 0 the CLI and the JSON schema may still change between minor
releases.

## 0.2.0 — first published release

Everything below existed before the package was published; 0.1.0 was never
released, and the number is bumped so the published history starts from a
version that reflects what the tool actually does.

### Audit

- Collectors for git submodules and LFS, Python declarations across six
  conventions, containers, imports, dependency resolution, build steps,
  environment variables, runtime assets, and the documented entrypoint.
- Cross-referencing: a missing file is reported as *fetched by this script*,
  *licence-gated at this provider*, or *referenced and fetched by nothing* —
  three findings that need different responses.
- A curated registry of licence gates, unreliable hosts, and packages a plain
  `pip install` cannot satisfy.

### Observation

- `syp trace` runs the entrypoint under a `sys.addaudithook` hook and records
  every path opened, module imported, binary spawned and host contacted.
  Runtime findings override inferred ones.
- `--target host|venv|image` runs probes inside the environment being audited,
  including `ldconfig` and the image's own `ENV`. Selecting an image also
  verifies GPU passthrough by starting a container with `--gpus all`.

### Honesty

- Requirements are scoped to the run being audited; `--entry` audits a
  different one. Files a script writes and reads back are not inputs.
- Present is not the same as correct: assets are checked for size, HTML
  interstitials, LFS pointers, magic bytes and published checksums.
- Fixes are classified local/network/script, and repository scripts are
  withheld unless `--allow-scripts`.
- An environment that cannot be inspected reports NOT VERIFIED rather than
  zero blockers.
- `.syp.toml` suppresses findings, and suppressions are counted out loud.

### Measurement

Validated against 19 public repositories. Systematic false-positive classes
found and removed there took the corpus from 1,315 blockers to 175, median 10
per repo, without losing real findings.

### Packaging

- Installs `syp` and `pyhole` (an alias, because an unrelated `syp` package
  exists on PyPI).
- Python 3.9+; `tomli` only on 3.10 and older.
- Tested on Linux, Windows and macOS against 3.9 and 3.13, and from the built
  wheel rather than the source tree.
