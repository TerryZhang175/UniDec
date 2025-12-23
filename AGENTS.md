# Repository Guidelines

## Project Structure
- `unidec/`: Python package (GUI + CLI). Entry points include `unidec/Launcher.py`, `unidec/GUniDec.py`, `python -m unidec` (CLI), and `python -m unidec.IsoDec`.
- `unidec/modules/`: core algorithms and GUI helpers.
- `unidec/UniDecImporter/`: file importers (mzML/mzXML/RAW/etc.).
- `unidec/src/` and `unidec/IsoDec/src_cmake/`: native C/C++ engines built via CMake/compile scripts.
- `unidec/bin/`: bundled executables, third-party runtime files, presets, and `unidec/bin/Example Data/` sample spectra.
- `PublicScripts/`: user-facing analysis scripts and workflows.

## Build, Test, and Development Commands
- Create env + deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Editable install: `pip install -e .`
- Run GUI (needs `wxPython`): `python -m unidec.Launcher`
- CLI smoke test: `python -m unidec "unidec/bin/Example Data/ADH.txt"`
- IsoDec smoke test: `python -m unidec.IsoDec "unidec/bin/Example Data/IsoDec/test2.txt" -precentroided`
- Build native engines (Linux): `bash unidec/src/compilelinux.sh` and `bash unidec/IsoDec/src_cmake/compilelinux.sh` (requires `cmake`, FFTW, HDF5).

## Coding Style & Naming Conventions
- Python: 4-space indentation; keep changes consistent with surrounding files and avoid repo-wide reformatting.
- Naming: `snake_case` for modules/functions, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Paths: quote paths containing spaces (e.g., `unidec/bin/Example Data/...`).
- Tooling: no enforced formatter/linter config in the repo; use local tools if desired but avoid formatting-only churn.

## Testing Guidelines
- Framework: `unittest` (see `unidec/UniDecImporter/ImportTests.py`), but it expects external test data paths and may not run out-of-the-box.
- Prefer lightweight smoke tests using `unidec/bin/Example Data/` for PR validation; add focused regression tests when fixing bugs.

## Commit & Pull Request Guidelines
- Commits in history are short, descriptive, and often past-tense (“Fixed…”, “Added…”, “Updated…”); keep subjects under ~72 chars and prefix a component when helpful (e.g., `IsoDec:`, `Docker:`).
- PRs: include what/why, how to test (exact commands + sample file), target OS(es), and screenshots for GUI changes. Avoid modifying `unidec/bin/` binaries unless required; call it out explicitly when you do.

