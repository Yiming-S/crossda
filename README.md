# crossda

Cross-session EEG classification and domain-adaptation pipelines for motor-imagery BCI.

`crossda` evaluates several strategies for transferring a classifier across recording
sessions of the same subject, under a leave-one-session-out protocol. It bundles four
pipeline families behind one CLI and a small Python API:

| Pipeline | Idea |
|----------|------|
| `MAP` | Merge & Adapt — supervised grid search over feature × DA × classifier, then merge all source sessions |
| `DWP` | Distance-Weighted Pooling — soft inverse-distance weighting of the full source pool |
| `MMP` | Minimum-distance Multi-source — CI-gated source selection (`merge_then_adapt` / `moe` combiners) |
| `BDP` | Bridge-Domain Proxy — proxy-tuning between a bridge set and a far set |

Features (`logvar` / `CSP` / `TS`), classifiers (`lda`, `svm_linear`, `svm_radial`, `lr`,
`el`, `elm`, `lgbm`, `catboost`, `mdm`) and domain-adaptation methods (`none`, `sa`, `tca`,
`pt`, `coral`) are all configurable via a method bank.

## Installation

```bash
git clone https://github.com/Yiming-S/CrossPython.git
cd CrossPython
pip install -e .            # core install — also pulls the da4bci backend from git
pip install -e ".[boost]"   # + lightgbm / catboost classifiers
pip install -e ".[dev]"     # + test tooling
```

Requires Python ≥ 3.10. The domain-adaptation backend
[`da4bci`](https://github.com/Yiming-S/DA4BCI-Python) is pulled automatically from its
git repository (it is not yet on PyPI).

## Quickstart

### Command line

```bash
crossda --help
crossda --mode smoke --datasets bnci004 --data-dir /path/to/EEG --result-dir ./out
crossda --pipelines MAP DWP --n-cores 4
crossda --config configs/examples/mmp_smoke.yaml
```

`crossda` with no arguments uses the packaged [`default.yaml`](src/crossda/default.yaml).
Copy it, set `data_dir` / `result_dir`, and pass it with `--config`. CLI flags override
the YAML.

### Python

```python
from crossda import Config, run

run(Config(
    datasets=["bnci004"],
    method_mode="smoke",
    data_dir="/path/to/EEG",
    result_dir="./out",
))
```

The individual pipeline functions are importable for custom experiments:

```python
from crossda import MAP, DWP, MMP, BDP, generate_method_bank, process_subject
```

## Datasets

`crossda` reads three motor-imagery datasets. `bnci004` is fetched automatically via
[MOABB](https://moabb.neurotechx.com/). `stieger2021` also uses the MOABB loader but
additionally needs the raw `.mat` files locally under `<data_dir>/MNE-Stieger2021-data/`
(used for subject discovery). `ma2020` is read from local `.cnt` files under
`<data_dir>/MNE-ma2020-data/`.

| Key | Dataset | Sessions |
|-----|---------|----------|
| `bnci004` | BNCI 2014-004 (9 subjects) | 5 |
| `stieger2021` | Stieger 2021 (62 subjects) | 6–11 |
| `ma2020` | Ma 2020 (25 subjects) | 15 |

## Outputs

For each subject and pipeline, a run writes a summary pickle, a detail pickle, and a
flattened per-session **roles** CSV (provenance: which sessions were train / target /
bridge / far, their weights and distances) to `result_dir`.

## Configuration

See [`src/crossda/default.yaml`](src/crossda/default.yaml) for the commonly-used options
(method bank mode, CV folds, MAP scoring mode, MMP bootstrap settings, parallelism, …); the
complete set of configurable fields is defined in [`config.py`](src/crossda/config.py).

## Citation

If you use `crossda` in academic work, please cite the associated paper (BibTeX TBD).

## License

[MIT](LICENSE) © Yiming Shen and David Degras
