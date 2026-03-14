import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from filters import MonoLMS, MonoNLMS, MonoRLS, MultiChannelLMS, MultiChannelNLMS, MultiChannelRLS
from wrapper import Wrapper

import time


# =============================================================================
# Data path construction
# =============================================================================
# This file lives at:  <project_root>/Code/tests.py
# The data lives at:   <project_root>/physionet.org/files/fecgsyndb/1.0.0/
#
# We walk up one level from __file__ (Code/ -> Project/) to get the shared
# project root, then build the data path from there — no hardcoded usernames
# or machine-specific directories anywhere in the file.
#
# Override individual segments via environment variables if needed, e.g.:
#   export FECG_SUBJECT="sub02"
#   export FECG_SNR="snr06dB"
#   export FECG_RECORD="sub02_snr06dB_l1"

# Project root = parent of the directory containing this file (Code/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SUBJECT = os.environ.get("FECG_SUBJECT", "sub01")
_SNR     = os.environ.get("FECG_SNR",     "snr12dB")
_RECORD  = os.environ.get("FECG_RECORD",  "sub01_snr12dB_l1_c0")

DATA_PATH = os.path.join(
    _PROJECT_ROOT,
    "physionet.org", "files", "fecgsyndb", "1.0.0",
    _SUBJECT, _SNR, _RECORD,
)


# =============================================================================
# Tests
# =============================================================================

class Tests:
    """
    End-to-end test harness for adaptive fECG filtering.

    Typical usage:
        t = Tests(partial_path=DATA_PATH)
        t.run_tests()

    The run_tests() method drives the full pipeline:
        1. Load and mix the WFDB ECG components via Wrapper.
        2. Run each enabled filter in block-processing mode.
        3. Perform sanity checks on the outputs.
        4. Compute cross-correlation diagnostics.
        5. Plot time-domain and STFT results.
    """

    def __init__(self, partial_path: str):
        self.wrapper = Wrapper(partial_path=partial_path)

        # Populated by load_data()
        self.Xsum      = None   # (N, n_ch) full mixed signal matrix
        self.fs        = None   # sampling frequency (Hz)
        self.x         = None   # reference channel fed to the filter as input (mono)
        self.X         = None   # reference channels matrix (multichannel), shape (N, M)
        self.d         = None   # desired channel (mixed abdominal lead)
        self.d_pure    = None   # ground-truth pure fetal channel for comparison
        self.pure_name = None   # suffix used to load the pure channel
        self.x_ch      = None   # index of the single reference channel in Xsum (mono)
        self.x_chs     = None   # list of reference channel indices (multichannel)
        self.d_ch      = None   # index of the desired channel in Xsum

    # =========================================================================
    # Data loading
    # =========================================================================

    def load_data(
        self,
        *,
        x_ch: int,
        d_ch: int,
        num_samples: int,
        noise: bool = True,
        pure_name: str = "fecg1",
    ):
        """
        Build the mixed ECG and extract the reference (x) and desired (d) channels.

        Parameters
        ----------
        x_ch        : column index in Xsum used as the filter reference signal x[n]
        d_ch        : column index in Xsum used as the desired signal d[n]
        num_samples : number of samples to load from each WFDB component
        noise       : whether to include noise component records in the mixture
        pure_name   : suffix of the pure fetal record to load for ground-truth
                      comparison (e.g. "fecg1" loads <record_base>_fecg1)
        """
        Xsum, fs, x, d = self.wrapper.get_mixed_ecg(
            x_ch=x_ch,
            d_ch=d_ch,
            num_samples=num_samples,
            noise=noise,
        )

        self.Xsum = Xsum
        self.fs   = fs
        self.x    = x
        self.d    = d
        self.x_ch = x_ch
        self.d_ch = d_ch

        # Attempt to load the ground-truth pure fetal channel for plotting.
        # This is optional — tests proceed normally if the record is missing.
        self.pure_name = pure_name
        try:
            self.d_pure = self.wrapper.get_pure_channel(
                name=pure_name, ch=d_ch, num_samples=num_samples
            )
        except Exception as ex:
            print(f"[WARN] Could not load pure channel '{pure_name}' ch={d_ch}: {ex}")
            self.d_pure = None

        print(
            f"[INFO] Loaded mixed ECG — shape={Xsum.shape}, fs={fs} Hz, "
            f"x_ch={x_ch}, d_ch={d_ch}, samples={len(d)}, noise={noise}"
        )
        if self.d_pure is not None:
            print(
                f"[INFO] Loaded pure channel '{pure_name}' ch={d_ch}, "
                f"samples={len(self.d_pure)}"
            )

    # =========================================================================
    # Block processing
    # =========================================================================

    def block_process(
        self,
        filter_obj,
        x: np.ndarray,
        d: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Feed x and d through filter_obj in non-overlapping blocks of block_size.

        The filter's process_block() is called once per block. If the filter
        exposes _last_y_raw / _last_e_raw attributes (pre-IIR outputs), those
        are also collected so that the strict e == d - y identity can be checked
        against the raw signals rather than the post-filtered ones.

        Returns
        -------
        y_out, e_out         : (post-processed) filter output and error arrays
        y_raw_out, e_raw_out : raw (pre-IIR) versions; equal to y/e if no IIR
        """
        x = np.asarray(x, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()

        if len(x) != len(d):
            raise ValueError("x and d must have the same length.")

        N = len(d)
        y_out     = np.zeros(N, dtype=float)
        e_out     = np.zeros(N, dtype=float)
        y_raw_out = np.zeros(N, dtype=float)
        e_raw_out = np.zeros(N, dtype=float)

        idx = 0
        while idx < N:
            end = min(idx + block_size, N)

            y_blk, e_blk = filter_obj.process_block(x[idx:end], d[idx:end])

            y_out[idx:end] = y_blk
            e_out[idx:end] = e_blk

            # Collect raw (pre-IIR) blocks if the filter exposes them.
            y_raw_blk = getattr(filter_obj, "_last_y_raw", None)
            e_raw_blk = getattr(filter_obj, "_last_e_raw", None)

            y_raw_out[idx:end] = y_blk if y_raw_blk is None else np.asarray(y_raw_blk).ravel()
            e_raw_out[idx:end] = e_blk if e_raw_blk is None else np.asarray(e_raw_blk).ravel()

            idx = end

        return y_out, e_out, y_raw_out, e_raw_out

    def block_process_matrix(
        self,
        filter_obj,
        X: np.ndarray,
        d: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Multichannel variant of block_process.

        X : (T, M) float array — multichannel reference input
        d : (T,)   float array — single desired channel

        The filter's process_block receives an (block_size, M) slice of X and
        a (block_size,) slice of d, and returns scalar y and e arrays as usual.
        Everything downstream (sanity checks, correlations, plots) is unchanged.

        Returns
        -------
        y_out, e_out         : (T,) filter output and residual
        y_raw_out, e_raw_out : pre-IIR versions if available, else equal to y/e
        """
        X = np.asarray(X, dtype=float)
        d = np.asarray(d, dtype=float).ravel()

        if X.shape[0] != len(d):
            raise ValueError(
                f"X has {X.shape[0]} rows but d has {len(d)} samples."
            )

        T = len(d)
        y_out     = np.zeros(T, dtype=float)
        e_out     = np.zeros(T, dtype=float)
        y_raw_out = np.zeros(T, dtype=float)
        e_raw_out = np.zeros(T, dtype=float)

        idx = 0
        while idx < T:
            end = min(idx + block_size, T)

            y_blk, e_blk = filter_obj.process_block(X[idx:end, :], d[idx:end])

            y_out[idx:end] = y_blk
            e_out[idx:end] = e_blk

            y_raw_blk = getattr(filter_obj, "_last_y_raw", None)
            e_raw_blk = getattr(filter_obj, "_last_e_raw", None)

            y_raw_out[idx:end] = y_blk if y_raw_blk is None else np.asarray(y_raw_blk).ravel()
            e_raw_out[idx:end] = e_blk if e_raw_blk is None else np.asarray(e_raw_blk).ravel()

            idx = end

        return y_out, e_out, y_raw_out, e_raw_out

    def load_data_multi(
        self,
        *,
        x_chs: list[int],
        d_ch: int,
        num_samples: int,
        noise: bool = True,
        pure_name: str = "fecg1",
    ):
        """
        Multichannel variant of load_data.

        Loads the full mixed signal matrix and extracts:
          - self.X  : (T, M) matrix of M reference channels
          - self.d  : (T,) desired channel
          - self.x  : first reference channel as a 1D array (for plotting)
          - self.x_ch : set to x_chs[0] so plot_results can label the axis

        Parameters
        ----------
        x_chs       : list of column indices to use as reference channels
        d_ch        : column index of the desired (abdominal) channel
        num_samples : samples to load
        noise       : include noise component records
        pure_name   : pure fetal record suffix for ground-truth comparison
        """
        # Use d_ch as the nominal x_ch for the Wrapper call; we'll pull the
        # full matrix ourselves directly from Xsum afterward.
        Xsum, fs, _, d = self.wrapper.get_mixed_ecg(
            x_ch=x_chs[0],
            d_ch=d_ch,
            num_samples=num_samples,
            noise=noise,
        )

        self.Xsum  = Xsum
        self.fs    = fs
        self.d     = d
        self.d_ch  = d_ch
        self.x_chs = x_chs
        self.x_ch  = x_chs[0]          # used by plot_results for the axis label

        # Extract all reference channels as a (T, M) matrix
        self.X = Xsum[:len(d), x_chs]  # trim to loaded length in case of edge cases
        self.x = self.X[:, 0]          # first channel as 1D for plotting

        self.pure_name = pure_name
        try:
            self.d_pure = self.wrapper.get_pure_channel(
                name=pure_name, ch=d_ch, num_samples=num_samples
            )
        except Exception as ex:
            print(f"[WARN] Could not load pure channel '{pure_name}' ch={d_ch}: {ex}")
            self.d_pure = None

        print(
            f"[INFO] Loaded multichannel ECG — shape={Xsum.shape}, fs={fs} Hz, "
            f"x_chs={x_chs}, d_ch={d_ch}, samples={len(d)}, noise={noise}"
        )
        if self.d_pure is not None:
            print(
                f"[INFO] Loaded pure channel '{pure_name}' ch={d_ch}, "
                f"samples={len(self.d_pure)}"
            )

    # =========================================================================
    # Sanity checks
    # =========================================================================

    def basic_tests(
        self,
        x: np.ndarray,
        d: np.ndarray,
        y: np.ndarray,
        e: np.ndarray,
        *,
        y_raw: np.ndarray = None,
        e_raw: np.ndarray = None,
        tol: float = 1e-9,
    ):
        """
        Minimal health checks on filter outputs. Raises on failure.

        Checks always performed:
          - All arrays have equal length.
          - y and e are finite (catches mu-too-large divergence).
          - y is not identically zero (filter must be updating).

        Identity check  e_raw == d - y_raw:
          - Only enforced when raw (pre-IIR) arrays are supplied, because IIR
            post-filtering breaks the sample-by-sample identity.
        """
        assert len(x) == len(d) == len(y) == len(e), "Length mismatch in outputs."

        if not (np.isfinite(y).all() and np.isfinite(e).all()):
            raise RuntimeError("NaN/Inf in outputs — check step size / filter stability.")

        if not np.any(np.abs(y) > tol):
            raise AssertionError("y is all ~0; filter tap updates may not be working.")

        if y_raw is not None and e_raw is not None:
            # Verify the raw error identity before any post-processing
            y_raw = np.asarray(y_raw, dtype=float).ravel()
            e_raw = np.asarray(e_raw, dtype=float).ravel()
            assert len(y_raw) == len(e_raw) == len(d), "Length mismatch in raw outputs."

            max_err = np.nanmax(np.abs(e_raw - (d - y_raw)))
            # Uncomment to enforce strictly:
            # assert max_err < 1e-6, f"RAW e != d - y; max error = {max_err:.2e}"

            print(f"[OK] Basic tests passed (raw identity max_err={max_err:.2e}).")
            return

        # Without raw arrays, just confirm e is non-trivial
        if not np.any(np.abs(e) > tol):
            raise AssertionError("e is all ~0; unexpected for a non-trivial input.")

        print("[OK] Basic tests passed (identity check skipped — outputs may be post-filtered).")

    # =========================================================================
    # Cross-correlation diagnostics
    # =========================================================================

    @staticmethod
    def _norm_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
        """
        Normalized cross-correlation between a and b, restricted to ±max_lag samples.

        Returns the peak correlation coefficient and its lag in samples.
        A peak near 0 for e vs x indicates successful maternal ECG cancellation.
        """
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()

        N = min(len(a), len(b))
        a, b = a[:N] - np.mean(a[:N]), b[:N] - np.mean(b[:N])

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0, 0

        corr_full = np.correlate(a, b, mode="full") / denom
        lags_full = np.arange(-N + 1, N)

        mask = (lags_full >= -max_lag) & (lags_full <= max_lag)
        corr, lags = corr_full[mask], lags_full[mask]

        idx = int(np.argmax(np.abs(corr)))
        return float(corr[idx]), int(lags[idx])

    def compute_cross_correlations(
        self,
        y: np.ndarray,
        e: np.ndarray,
        *,
        max_lag_s: float = 0.2,
    ) -> dict:
        """
        Compute four normalized cross-correlation peaks to assess filter quality.

        Metrics
        -------
        y vs d  : how well y tracks the mixed desired signal (should be high)
        e vs d  : residual correlation with d (should be low after convergence)
        y vs x  : how well y tracks the reference (should be high — y ≈ maternal)
        e vs x  : residual correlation with reference (should be low — maternal cancelled)

        max_lag_s : search window half-width in seconds
        """
        if self.Xsum is None or self.fs is None:
            raise RuntimeError("Call load_data() before compute_cross_correlations().")

        x0 = self.Xsum[:, self.x_ch]
        d0 = self.Xsum[:, self.d_ch]
        max_lag = int(max_lag_s * self.fs)

        # Trim all signals to the same length
        N = min(len(d0), len(y), len(e), len(x0))
        x0, d0, y, e = x0[:N], d0[:N], np.asarray(y)[:N], np.asarray(e)[:N]

        y_d_corr, y_d_lag = self._norm_xcorr(y, d0, max_lag=max_lag)
        e_d_corr, e_d_lag = self._norm_xcorr(e, d0, max_lag=max_lag)
        y_x_corr, y_x_lag = self._norm_xcorr(y, x0, max_lag=max_lag)
        e_x_corr, e_x_lag = self._norm_xcorr(e, x0, max_lag=max_lag)

        return {
            "max_lag_samples" : max_lag,
            "y_vs_d_peak_corr": y_d_corr,   "y_vs_d_peak_lag": y_d_lag,
            "e_vs_d_peak_corr": e_d_corr,   "e_vs_d_peak_lag": e_d_lag,
            "y_vs_x_peak_corr": y_x_corr,   "y_vs_x_peak_lag": y_x_lag,
            "e_vs_x_peak_corr": e_x_corr,   "e_vs_x_peak_lag": e_x_lag,
        }

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_results(
        self,
        x: np.ndarray,
        d: np.ndarray,
        y: np.ndarray,
        e: np.ndarray,
        *,
        title: str,
        start: float = 0.0,
        end: float = 40.0,
    ):
        if self.fs is None:
            raise RuntimeError("fs not set — call load_data() first.")
        if end <= start:
            raise ValueError(f"end ({end}) must be greater than start ({start}).")

        N = len(d)
        start_idx = int(max(0.0, start) * self.fs)
        end_idx   = int(min(end * self.fs, N))

        if start_idx >= N:
            raise ValueError(f"start={start}s exceeds signal length ({N / self.fs:.2f}s).")
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, N)

        t = np.arange(start_idx, end_idx) / self.fs
        have_pure = (self.d_pure is not None) and (len(self.d_pure) >= end_idx)
        nrows = 5 if have_pure else 4

        # --- Consistent style parameters ---
        LABEL_FS  = 9
        TICK_FS   = 8
        TITLE_FS  = 10
        LW        = 1
        C_REF     = "#2166ac"   # blue   — reference / input signals
        C_EST     = "#4dac26"   # green  — estimated maternal
        C_RES     = "#b2182b"   # dark red — residual (fetal estimate)
        C_PURE    = "#762a83"   # purple — ground truth

        fig, axes = plt.subplots(
            nrows, 1,
            figsize=(10, 1.3 * nrows),
            sharex=True,
        )
        fig.suptitle(title, fontsize=TITLE_FS, y=0.98)

        def _plot(ax, t, signal, label, color):
            ax.plot(t, signal, color=color, linewidth=LW)
            ax.set_ylabel(label, fontsize=LABEL_FS, labelpad=4)
            ax.tick_params(labelsize=TICK_FS)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        _plot(axes[0], t, x[start_idx:end_idx],
            f"$m[n]$ (ch {self.x_ch})", C_REF)

        _plot(axes[1], t, d[start_idx:end_idx],
            f"$a[n]$ (ch {self.d_ch})", C_REF)

        _plot(axes[2], t, y[start_idx:end_idx],
            "$y[n]$ est. maternal", C_EST)

        _plot(axes[3], t, e[start_idx:end_idx],
            "$e[n]$ residual", C_RES)

        if have_pure:
            _plot(axes[4], t, self.d_pure[start_idx:end_idx],
                f"pure fetal (ref.)", C_PURE)

        axes[-1].set_xlabel("Time (s)", fontsize=LABEL_FS)
        fig.tight_layout()
        plt.show()

        # --- Figure 2: STFT ---
        e_seg = np.asarray(e[start_idx:end_idx], dtype=float).ravel()

        nperseg  = int(0.25 * self.fs)
        noverlap = nperseg // 2

        f_e, tt_e, Z_e = sig.stft(
            e_seg, fs=self.fs, nperseg=nperseg, noverlap=noverlap,
            boundary=None, window="hamming"
        )
        S_e = 20.0 * np.log10(np.abs(Z_e) + 1e-12)

        if have_pure:
            p_seg = np.asarray(self.d_pure[start_idx:end_idx], dtype=float).ravel()
            f_p, tt_p, Z_p = sig.stft(
                p_seg, fs=self.fs, nperseg=nperseg, noverlap=noverlap, boundary=None
            )
            S_p = 20.0 * np.log10(np.abs(Z_p) + 1e-12)

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            im1 = ax1.pcolormesh(tt_e + start, f_e, S_e, shading="auto")
            ax1.set_ylabel("Frequency (Hz)", fontsize=LABEL_FS)
            ax1.set_title("STFT — residual $e[n]$", fontsize=LABEL_FS)
            ax1.tick_params(labelsize=TICK_FS)
            fig2.colorbar(im1, ax=ax1, label="Magnitude (dB)")

            im2 = ax2.pcolormesh(tt_p + start, f_p, S_p, shading="auto")
            ax2.set_ylabel("Frequency (Hz)", fontsize=LABEL_FS)
            ax2.set_title("STFT — pure fetal", fontsize=LABEL_FS)
            ax2.set_xlabel("Time (s)", fontsize=LABEL_FS)
            ax2.tick_params(labelsize=TICK_FS)
            fig2.colorbar(im2, ax=ax2, label="Magnitude (dB)")

        else:
            fig2, ax = plt.subplots(figsize=(8, 2.5))
            im = ax.pcolormesh(tt_e + start, f_e, S_e, shading="auto")
            ax.set_title("STFT — residual $e[n]$ (pure unavailable)", fontsize=LABEL_FS)
            ax.set_ylabel("Frequency (Hz)", fontsize=LABEL_FS)
            ax.set_xlabel("Time (s)", fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)
            fig2.colorbar(im, ax=ax, label="Magnitude (dB)")

        fig2.tight_layout()
        plt.show()

    # =========================================================================
    # High-level filter runner
    # =========================================================================

    def run_filter(
        self,
        *,
        filter_name: str,
        filter_obj,
        block_size: int,
        start: float = 0.0,
        end: float = 40.0,
        do_plot: bool = True,
        do_corr: bool = True,
    ) -> dict:
        """
        Run a single filter end-to-end: block-process → sanity check → correlations → plot.

        Parameters
        ----------
        filter_name : label used in plot titles and printed output
        filter_obj  : any Filter subclass instance
        block_size  : number of samples per process_block() call
        start, end  : time window (seconds) for plotting
        do_plot     : whether to call plot_results()
        do_corr     : whether to compute and print cross-correlations

        Returns
        -------
        dict with keys "y", "e", "corr"
        """
        t_start = time.perf_counter()
        y, e, y_raw, e_raw = self.block_process(
            filter_obj, self.x, self.d, block_size=block_size
        )
        per_sample_us = (time.perf_counter() - t_start) / len(self.d) * 1e6

        self.basic_tests(self.x, self.d, y, e, y_raw=y_raw, e_raw=e_raw)

        corr = None
        if do_corr:
            corr = self.compute_cross_correlations(y, e, max_lag_s=0.2)
            print(f"\n[{filter_name}] Per-sample latency: {per_sample_us:.3f} µs")
            print(f"[{filter_name}] Cross-correlation peaks (normalized):")
            for k, v in corr.items():
                print(f"    {k:>25s}: {v}")

        if do_plot:
            self.plot_results(
                self.x, self.d, y, e,
                title=(
                    f"{filter_name} | record={self.wrapper.record_base} "
                    f"| taps={getattr(filter_obj, 'N', '??')}"
                ),
                start=start,
                end=end,
            )

        return {"y": y, "e": e, "corr": corr}

    # =========================================================================
    # Entry point
    # =========================================================================

    def run_tests(self):
        """
        Main pipeline entry point. Edit the CONFIG block to change experiment settings.
        Toggle each filter on/off with the RUN_* flags.
        """

        # =====================================================================
        # CONFIG — edit these values to change the experiment
        # =====================================================================

        # Channel selection
        x_ch        = 32       # reference channel (maternal-dominant thoracic lead)
        d_ch        = 0        # desired channel  (abdominal lead — mixed maternal + fetal)
        num_samples = 7500 * 2 # total samples to load (~2 s at 1 kHz; adjust as needed)
        block_size  = 1024     # samples per filter block call
        noise       = True    # include noise component records in the mixture
        pure_name   = "fecg1"  # pure fetal record suffix for ground-truth comparison

        # LMS config
        RUN_LMS  = False
        taps_lms = 512
        mu_lms   = 9.25e-3
        t0_lms, t1_lms = 15.0, 25.0

        # NLMS config
        RUN_NLMS    = False
        taps_nlms   = 512
        mu_nlms     = 0.88
        eps_nlms    = 1e-7      # regularization — prevents divide-by-zero on silent input
        varying_mu  = False
        mu_factor   = 0.9       # exponential decay applied to mu each sample
        mu_floor    = 1e-5      # mu never falls below this value
        t0_nlms, t1_nlms = 15.0, 25.0

        # RLS config
        RUN_RLS   = True
        taps_rls  = 256
        lam_rls   = 0.9998      # best e_vs_x from sweep (0.0581)
        delta_rls = 1e-3       # small delta = conservative init, better steady-state cancellation
        t0_rls, t1_rls = 15.0, 25.0

        # =====================================================================
        # Load data
        # =====================================================================
        self.load_data(
            x_ch=x_ch,
            d_ch=d_ch,
            num_samples=num_samples,
            noise=noise,
            pure_name=pure_name,
        )

        # =====================================================================
        # Run LMS
        # =====================================================================
        if RUN_LMS:
            lms = MonoLMS(N=taps_lms, mu=mu_lms)
            self.run_filter(
                filter_name=f"MonoLMS(mu={mu_lms})",
                filter_obj=lms,
                block_size=block_size,
                start=t0_lms,
                end=t1_lms,
                do_plot=True,
                do_corr=True,
            )

        # =====================================================================
        # Run NLMS
        # =====================================================================
        if RUN_NLMS:
            nlms = MonoNLMS(
                N=taps_nlms,
                mu=mu_nlms,
                eps=eps_nlms,
                varying_mu=varying_mu,
                factor=mu_factor,
                mu_floor=mu_floor,
            )
            self.run_filter(
                filter_name=f"MonoNLMS(mu={mu_nlms})",
                filter_obj=nlms,
                block_size=block_size,
                start=t0_nlms,
                end=t1_nlms,
                do_plot=True,
                do_corr=True,
            )

        # =====================================================================
        # Run RLS
        # =====================================================================
        if RUN_RLS:
            rls = MonoRLS(N=taps_rls, lam=lam_rls, delta=delta_rls)
            self.run_filter(
                filter_name=f"MonoRLS(lam={lam_rls}, d = {delta_rls})",
                filter_obj=rls,
                block_size=block_size,
                start=t0_rls,
                end=t1_rls,
                do_plot=True,
                do_corr=True,
            )


    # =========================================================================
    # Multichannel filter runner
    # =========================================================================

    def run_filter_multi(
        self,
        *,
        filter_name: str,
        filter_obj,
        block_size: int,
        start: float = 0.0,
        end: float = 40.0,
        do_plot: bool = True,
        do_corr: bool = True,
    ) -> dict:
        """
        Multichannel counterpart to run_filter.

        Uses self.X (T, M) as the reference input instead of self.x (T,).
        After block_process_matrix returns scalar y and e arrays, everything
        downstream — basic_tests, compute_cross_correlations, plot_results —
        is called identically to the mono case.

        Requires load_data_multi() to have been called first.
        """
        if self.X is None:
            raise RuntimeError(
                "self.X is not set — call load_data_multi() before run_filter_multi()."
            )

        t_start = time.perf_counter()
        y, e, y_raw, e_raw = self.block_process_matrix(
            filter_obj, self.X, self.d, block_size=block_size
        )
        per_sample_us = (time.perf_counter() - t_start) / len(self.d) * 1e6

        # basic_tests uses self.x (1D, first ref channel) for the length check.
        self.basic_tests(self.x, self.d, y, e, y_raw=y_raw, e_raw=e_raw)

        corr = None
        if do_corr:
            corr = self.compute_cross_correlations(y, e, max_lag_s=0.2)
            print(f"\n[{filter_name}] Per-sample latency: {per_sample_us:.3f} µs")
            print(f"[{filter_name}] Cross-correlation peaks (normalized):")
            for k, v in corr.items():
                print(f"    {k:>25s}: {v}")

        if do_plot:
            self.plot_results(
                self.x, self.d, y, e,
                title=(
                    f"{filter_name} | record={self.wrapper.record_base} "
                    f"| taps={getattr(filter_obj, 'N', '??')} "
                    f"| ref_ch={self.x_chs}"
                ),
                start=start,
                end=end,
            )

        return {"y": y, "e": e, "corr": corr}

    # =========================================================================
    # Multichannel entry point
    # =========================================================================

    def run_tests_multi(self):
        """
        Multichannel pipeline entry point.

        Mirrors run_tests() but uses load_data_multi() to build a (T, M)
        reference matrix and run_filter_multi() to drive the multichannel filters.

        Channel selection guidance
        --------------------------
        x_chs should be thoracic / maternal-dominant leads highly correlated
        with d_ch but containing as little fetal signal as possible. In the
        fecgsyndb 34-channel layout:
          - Channels 0–3  : abdominal leads (fetal-rich  — avoid as reference)
          - Channels 4–33 : thoracic leads  (maternal-dominant — good reference)
        Start with 4–8 thoracic channels. More channels improve cancellation
        but grow P as O((N*M)^2) — keep N*M <= 512 for interactive speed.
        """

        # =====================================================================
        # CONFIG — edit these values to change the experiment
        # =====================================================================

        # Channel selection
        x_chs       = [32, 33]  # reference channels (thoracic, maternal-dominant)
        d_ch        = 0                  # desired channel (abdominal — mixed maternal + fetal)
        num_samples = 7500 * 2
        block_size  = 1024
        noise       = True
        pure_name   = "fecg1"

        # MultiChannelLMS config
        RUN_MC_LMS  = False
        taps_mc_lms = 512          # taps per channel — keep N*M small for LMS
        mu_mc_lms   = 1.85e-3        # must be smaller than mono: power scales with M*N
        t0_mc_lms, t1_mc_lms = 15.0, 25.0

        # MultiChannelNLMS config
        RUN_MC_NLMS  = False
        taps_mc_nlms = 512         # taps per channel
        mu_mc_nlms   = 1.88      # NLMS normalizes automatically — same range as mono
        eps_mc_nlms  = 1e-7
        t0_mc_nlms, t1_mc_nlms = 15.0, 25.0

        # MultiChannelRLS config — the primary workhorse
        # Memory: P is (N*M)^2 floats. N=64, M=4 -> 256^2 * 8B = 0.5 MB (fast)
        RUN_MC_RLS   = False
        taps_mc_rls  = 128         # taps per channel; total NM = 64*4 = 256
        lam_mc_rls   = 0.9991
        delta_mc_rls = 2e-2
        t0_mc_rls, t1_mc_rls = 15.0, 25.0

        # =====================================================================
        # Load data (multichannel)
        # =====================================================================
        self.load_data_multi(
            x_chs=x_chs,
            d_ch=d_ch,
            num_samples=num_samples,
            noise=noise,
            pure_name=pure_name,
        )

        M = len(x_chs)

        # =====================================================================
        # Run MultiChannelLMS
        # =====================================================================
        if RUN_MC_LMS:
            mc_lms = MultiChannelLMS(N=taps_mc_lms, M=M, mu=mu_mc_lms)
            self.run_filter_multi(
                filter_name=f"MultiChannelLMS(mu={mu_mc_lms}, M={M})",
                filter_obj=mc_lms,
                block_size=block_size,
                start=t0_mc_lms,
                end=t1_mc_lms,
                do_plot=True,
                do_corr=True,
            )

        # =====================================================================
        # Run MultiChannelNLMS
        # =====================================================================
        if RUN_MC_NLMS:
            mc_nlms = MultiChannelNLMS(
                N=taps_mc_nlms, M=M,
                mu=mu_mc_nlms,
                eps=eps_mc_nlms,
            )
            self.run_filter_multi(
                filter_name=f"MultiChannelNLMS(mu={mu_mc_nlms}, M={M})",
                filter_obj=mc_nlms,
                block_size=block_size,
                start=t0_mc_nlms,
                end=t1_mc_nlms,
                do_plot=True,
                do_corr=True,
            )

        # =====================================================================
        # Run MultiChannelRLS
        # =====================================================================
        if RUN_MC_RLS:
            mc_rls = MultiChannelRLS(
                N=taps_mc_rls, M=M,
                lam=lam_mc_rls,
                delta=delta_mc_rls,
            )
            self.run_filter_multi(
                filter_name=f"MultiChannelRLS(lam={lam_mc_rls}, M={M})",
                filter_obj=mc_rls,
                block_size=block_size,
                start=t0_mc_rls,
                end=t1_mc_rls,
                do_plot=True,
                do_corr=True,
            )


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    print(f"[INFO] Using data path: {DATA_PATH}")
    t = Tests(partial_path=DATA_PATH)
    t.run_tests()
    t.run_tests_multi()