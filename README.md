# Non-Invasive Fetal ECG Extraction

Companion code for the report *"Non-Invasive Fetal ECG Extraction: A Comparison of Time-Domain Adaptive Filtering Methods"*, written for ELECT\_ENG 495 — Cardiovascular Instrumentation at Northwestern University (March 2026).

> **Note:** This repository reflects a version of the work extended after the course submission to include a **PCA-based extraction method** as an additional baseline. The original report covered only adaptive filtering (LMS, NLMS, RLS and their multichannel variants). The PCA section and the associated code in `filters.py` and `tests.py` were added post-submission.

---

## What it does

The code extracts a fetal ECG (fECG) signal from noisy abdominal electrode recordings in which the maternal ECG (mECG) is roughly an order of magnitude larger. Seven configurations are implemented and compared:

| Method | Channels used | Reference needed |
|---|---|---|
| Mono LMS | 1 thoracic ref + 1 abdominal | Yes |
| Mono NLMS | 1 thoracic ref + 1 abdominal | Yes |
| Mono RLS | 1 thoracic ref + 1 abdominal | Yes |
| Multi-Channel LMS | 2 thoracic refs + 1 abdominal | Yes |
| Multi-Channel NLMS | 2 thoracic refs + 1 abdominal | Yes |
| Multi-Channel RLS | 2 thoracic refs + 1 abdominal | Yes |
| **PCA** | **3 abdominal channels only** | **No** |

---

## Data

All experiments use the [Fetal ECG Synthetic Database (fecgsyndb)](https://physionet.org/content/fecgsyndb/1.0.0/) from PhysioNet. The database is not included in this repository and must be downloaded separately. Place it so that the path structure is:

```
<project_root>/physionet.org/files/fecgsyndb/1.0.0/sub01/snr12dB/...
```

All results in the report use subject `sub01`, SNR `12 dB`, lead set `l1`, case `c0`, with noise included, at 250 Hz.

---

## File overview

### `wrapper.py`
Loads and mixes WFDB component records (mECG, fECG, noise) into a single 34-channel signal matrix. Exposes `get_mixed_ecg()` for the full mixture and `get_pure_channel()` for ground-truth component extraction.

### `filters.py`
All filter implementations as Python classes:

- **`Filter`** — abstract base class for mono FIR adaptive filters; handles the sample-by-sample delay buffer and block loop.
- **`MonoLMS`**, **`MonoNLMS`**, **`MonoRLS`** — mono adaptive filters.
- **`MultiFilter`** — abstract base class for multichannel filters; manages the flattened NM supervector.
- **`MultiChannelLMS`**, **`MultiChannelNLMS`**, **`MultiChannelRLS`** — multichannel variants.
- **`IIRPostFilter`** — mixin that adds optional stateful IIR post-filtering to any filter output.
- **`PCA_Filter`** *(added post-submission)* — batch SVD-based maternal cancellation operating on abdominal channels only. Includes SVD sign correction, skewness-based polarity check, and a geometric-mean amplitude gain estimator.

### `tests.py`
End-to-end test harness. Loads data, runs filters in block-processing mode, computes normalized cross-correlation metrics, and produces time-domain and STFT plots. Entry points:

```python
t = Tests(partial_path=DATA_PATH)
t.run_tests()        # mono adaptive filters
t.run_tests_multi()  # multichannel adaptive filters
t.run_tests_pca()    # PCA
```

---

## Dependencies

```
numpy
scipy
matplotlib
wfdb
pandas
```

Install with:

```bash
pip install numpy scipy matplotlib wfdb pandas
```

---

## Quick start

```python
from tests import Tests

DATA_PATH = "physionet.org/files/fecgsyndb/1.0.0/sub01/snr12dB/sub01_snr12dB_l1_c0"

t = Tests(partial_path=DATA_PATH)
t.run_tests_pca()
```

The data path can also be overridden via environment variables:

```bash
export FECG_SUBJECT="sub02"
export FECG_SNR="snr06dB"
export FECG_RECORD="sub02_snr06dB_l1_c0"
```
