# Contributing to Dendros

Thank you for your interest in contributing to Dendros!  We welcome bug
reports, feature requests, documentation improvements, and code contributions.

---

## Table of Contents

1. [Development environment setup](#development-environment-setup)
2. [Running tests](#running-tests)
3. [Formatting and linting](#formatting-and-linting)
4. [Proposing changes](#proposing-changes)

---

## Development environment setup

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/<your-username>/dendros.git
   cd dendros
   ```

2. **Create a virtual environment** (Python ≥ 3.8):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode** with development extras:

   ```bash
   pip install -e ".[dev,pandas,docs]"
   ```

   This installs:
   - `dendros` itself (editable)
   - `pytest` and `pytest-cov` for testing
   - `pandas` for optional table output
   - `sphinx`, `sphinx-rtd-theme`, `myst-parser`, `nbsphinx` for docs

---

## Running tests

```bash
pytest
```

To also measure code coverage:

```bash
pytest --cov=dendros --cov-report=term-missing
```

Tests live in the `tests/` directory and use small synthetic HDF5 files
generated in temporary directories – no real Galacticus output data is needed.

---

## Formatting and linting

We follow standard Python style guidelines (PEP 8).  We recommend using
[`black`](https://black.readthedocs.io) for auto-formatting and
[`ruff`](https://docs.astral.sh/ruff/) for linting:

```bash
pip install black ruff
black src/ tests/
ruff check src/ tests/
```

Type annotations are encouraged throughout the codebase.

---

## Building the documentation

```bash
cd docs
make html
# open _build/html/index.html in your browser
```

---

## Proposing changes

1. **Open an issue** first to discuss the change you would like to make,
   especially for larger features or breaking API changes.

2. **Create a branch** from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Implement your changes**, adding or updating tests as needed.

4. **Run the test suite** to make sure everything passes:

   ```bash
   pytest
   ```

5. **Open a pull request** against `galacticusorg/dendros:main`.
   Describe _what_ changed and _why_; link the relevant issue if one exists.

6. A maintainer will review the PR.  Please be responsive to feedback – we aim
   to keep the review cycle short.

---

## Code of Conduct

Please be respectful and constructive in all interactions.  We follow the
[Contributor Covenant](https://www.contributor-covenant.org) code of conduct.
