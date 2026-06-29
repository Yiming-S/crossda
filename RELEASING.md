# Releasing crossda

crossda publishes to PyPI via GitHub Actions **Trusted Publishing** (see
`.github/workflows/release.yml`). No API tokens are stored in the repo.

## ⚠️ da4bci must be on PyPI first

crossda depends on `da4bci`, currently pinned as a git URL in `pyproject.toml`:

```
da4bci @ git+https://github.com/Yiming-S/DA4BCI-Python.git
```

**PyPI rejects packages with direct-URL dependencies**, so before releasing crossda:

1. Publish **da4bci** to PyPI first (see its `RELEASING.md`).
2. In `pyproject.toml`, change the dependency from the git URL to a version spec:
   ```
   "da4bci>=0.1.0",
   ```
3. Then release crossda.

## One-time setup

1. The first upload claims the project name `crossda` on PyPI.
2. On PyPI → **Publishing** → add a Trusted Publisher:
   - Owner: `Yiming-S`, Repository: `CrossPython`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In the GitHub repo → **Settings → Environments** → create an environment named
   `pypi`.

## Each release

1. Confirm the da4bci dependency is a version spec (not a git URL).
2. Bump `version` in `pyproject.toml` and `__version__` in `src/crossda/__init__.py`.
3. Update `CHANGELOG.md`.
4. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
5. The `release.yml` workflow builds and publishes to PyPI.
