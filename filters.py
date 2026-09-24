import numpy as np
from abc import ABC, abstractmethod

import scipy.signal as sig
import pandas as pd


# IIR Post-Filter Mixin

class IIRPostFilter:
    """
    Mixin that adds stateful IIR post-filtering to any Filter subclass.
    Call _init_iir() in __init__ and _apply_iir() at the end of process_block.
    """

    def _init_iir(
        self,
        iir_enable: bool,
        iir_on: str,
        taps_path: str,
    ):
        self.iir_enable = bool(iir_enable)
        self.iir_on = str(iir_on)

        # Get coefficients
        if self.iir_enable:
            if not taps_path:
                raise ValueError("taps_path must be provided when iir_enable=True.")
            arr = pd.read_csv(taps_path, header=None).to_numpy()
            self.b = arr[:, 0].ravel()
            self.a = arr[:, 1].ravel()
            L = max(len(self.a), len(self.b)) - 1
            self.zi_y = np.zeros(L, dtype=float)
            self.zi_e = np.zeros(L, dtype=float)

    def _iir_filter(self, x: np.ndarray, which: str) -> np.ndarray:
        """Apply  IIR filter to a block. which: 'y' or 'e'."""
        if which == "y":
            out, self.zi_y = sig.lfilter(self.b, self.a, x, zi=self.zi_y)
        elif which == "e":
            out, self.zi_e = sig.lfilter(self.b, self.a, x, zi=self.zi_e)
        else:
            raise ValueError(f"which must be 'y' or 'e', got '{which}'.")
        return out

    def _apply_iir(
        self, y_block: np.ndarray, e_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Post-filter y and/or e blocks according to iir_on setting."""
        if not self.iir_enable:
            return y_block, e_block
        if self.iir_on in ("e", "both"):
            e_block = self._iir_filter(e_block, which="e")
        if self.iir_on in ("y", "both"):
            y_block = self._iir_filter(y_block, which="y")
        return y_block, e_block


# Abstract Base Filter
class Filter(ABC):
    """
    Base class for all adaptive FIR filters.

    Subclasses must implement _compute_step(d_n) which:
      - assumes self.x_buf is already updated for sample n
      - computes self.y and self.e
      - updates self.taps
    """

    def __init__(self, N: int):
        self.N = int(N)
        self.taps = np.zeros(self.N, dtype=float)
        self.x_buf = np.zeros(self.N, dtype=float)
        self.y = 0.0
        self.e = 0.0

    @abstractmethod
    def _compute_step(self, d_n: float):
        """Process a single sample: update taps, set self.y and self.e."""
        pass

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        
        # Ensure proper shape 
        x_block = np.asarray(x_block, dtype=float).ravel()
        d_block = np.asarray(d_block, dtype=float).ravel()

        if len(x_block) != len(d_block):
            raise ValueError("x_block and d_block must have the same length.")

        # Initiate outputs
        y_block = np.zeros_like(d_block, dtype=float)
        e_block = np.zeros_like(d_block, dtype=float)

        # Loop through samples
        for n in range(len(d_block)):

            # Shift buffer and update latest sample
            self.x_buf[1:] = self.x_buf[:-1]
            self.x_buf[0] = x_block[n]

            # Update filters and latest y and e internally
            self._compute_step(float(d_block[n]))

            # Populate output
            y_block[n] = self.y
            e_block[n] = self.e

        return y_block, e_block


# Standard LMS

class MonoLMS(Filter):
    def __init__(self, N: int, mu: float = 0.01):
        """
        N  : number of FIR taps
        mu : LMS step size
        """
        super().__init__(N)
        self.mu = float(mu)

    def _compute_step(self, d_n: float):

        # Dot product for computing output
        self.y = float(self.taps @ self.x_buf)

        # Difference for computing error
        self.e = d_n - self.y

        # Update equation
        self.taps += self.mu * self.e * self.x_buf


# NLMS

class MonoNLMS(IIRPostFilter, Filter):
    def __init__(
        self,
        N: int,
        mu: float = 0.01,
        eps: float = 1e-5,
        *,
        varying_mu: bool = True,
        mu_floor: float = 1e-5,
        factor: float = 0.8,
        iir_enable: bool = False,
        iir_on: str = "e",
        taps_path: str = "",
    ):
        """
        N         : number of FIR taps
        mu        : initial NLMS step size
        eps       : regularization term
        varying_mu: exponentially decay mu each sample
        mu_floor  : minimum mu when varying_mu is enabled
        factor    : decay factor applied to mu each sample
        iir_enable: enable IIR post-filtering on outputs
        iir_on    : which output(s) to filter — "e", "y", or "both"
        taps_path : CSV with IIR b coefficients in col 0, a in col 1
        """
        # Call the right parent constructor
        Filter.__init__(self, N)

        self.mu = float(mu)
        self.eps = float(eps)
        self.varying_mu = bool(varying_mu)
        self.mu_floor = float(mu_floor)
        self.factor = float(factor)

        # Initialize the IIR object 
        self._init_iir(iir_enable, iir_on, taps_path)

    def _compute_step(self, d_n: float):

        # Dot product
        self.y = float(self.taps @ self.x_buf)

        # Error
        self.e = d_n - self.y

        # Normalized update with margin for safe division
        denom = self.eps + float(self.x_buf @ self.x_buf)
        self.taps += (self.mu / denom) * self.e * self.x_buf

        # Update mu
        if self.varying_mu:
            self.mu = max(self.mu * self.factor, self.mu_floor)

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        
        # Get output blocks
        y_block, e_block = super().process_block(x_block, d_block)

        # Return filtered version (_apply_iir only filters when the iir_enable flag is True)
        return self._apply_iir(y_block, e_block)


# RLS

class MonoRLS(IIRPostFilter, Filter):
    def __init__(
        self,
        N: int,
        lam: float = 0.999,
        delta: float = 1e-2,
        *,
        iir_enable: bool = False,
        iir_on: str = "e",
        taps_path: str = "",
    ):
        """
        N     : number of FIR taps
        lam   : forgetting factor (0.99–0.9999 typical); lower = faster tracking
        delta : initial inverse-correlation matrix scale — P = (1/delta) * I
        iir_enable: enable IIR post-filtering on outputs
        iir_on    : which output(s) to filter — "e", "y", or "both"
        taps_path : CSV with IIR b coefficients in col 0, a in col 1
        """

        # Initialize the proper parent class
        Filter.__init__(self, N)

        self.lam = float(lam)
        self.lam_inv = 1.0 / self.lam
        self.P = (1.0 / delta) * np.eye(N, dtype=float)

        # Initialize the iir filter
        self._init_iir(iir_enable, iir_on, taps_path)

    def _compute_step(self, d_n: float):

        # Recursively  pre-compute P(x), the inverse of R(x), the weighted input correlation matrix
        Px = self.P @ self.x_buf

        # Get the kalman gain
        denom = self.lam + float(self.x_buf @ Px)
        k = Px / denom

        # Filter and produce outputs
        self.y = float(self.taps @ self.x_buf)
        self.e = d_n - self.y

        # Use the classic RLS update rules
        self.taps += k * self.e
        self.P = self.lam_inv * (self.P - np.outer(k, Px))

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        
        y_block, e_block = super().process_block(x_block, d_block)
        return self._apply_iir(y_block, e_block)
    

# =============================================================================
# Abstract Base — Multichannel
# =============================================================================

class MultiFilter(ABC):
    """
    Base class for multichannel adaptive FIR filters.

    Each of the M reference channels gets its own bank of N taps, giving a
    total of N*M parameters. Internally, these are stored as a flat supervector
    so the math stays identical to the mono case and the filter just operates on
    a longer x_buf formed by concatenating all M delay lines.

    Shape conventions
    -----------------
    x_block : (T, M)  — T samples, M reference channels
    d_block : (T,)    — single desired channel (one abdominal lead)
    taps    : (N*M,)  — flattened weight vector  [w_ch0 | w_ch1 | ... | w_chM]
    x_buf   : (N*M,)  — flattened delay buffer   [x0_delays | x1_delays | ...]

    Subclasses implement _compute_step(d_n) exactly as in the mono case.
    The base class handles buffer management and block looping.
    """

    def __init__(self, N: int, M: int):
        self.N  = int(N)   # taps per channel
        self.M  = int(M)   # number of reference channels
        self.NM = self.N * self.M

        # Flat supervector: [ch0_delay0..ch0_delayN-1 | ch1_delay0 .. | ...]
        self.taps  = np.zeros(self.NM, dtype=float)
        self.x_buf = np.zeros(self.NM, dtype=float)
        self.y = 0.0
        self.e = 0.0

    @abstractmethod
    def _compute_step(self, d_n: float):
        """Process one sample. x_buf is already updated before this is called."""
        pass

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        x_block : (T, M) float array — multichannel reference input
        d_block : (T,)   float array — desired signal

        Returns
        -------
        y_block : (T,) estimated maternal component
        e_block : (T,) residual (target fetal ECG)
        """
        x_block = np.asarray(x_block, dtype=float)
        d_block = np.asarray(d_block, dtype=float).ravel()

        if x_block.ndim == 1:
            # Gracefully handle accidental mono input
            x_block = x_block[:, np.newaxis]

        T = d_block.shape[0]
        if x_block.shape[0] != T:
            raise ValueError(
                f"x_block has {x_block.shape[0]} rows but d_block has {T} samples."
            )
        if x_block.shape[1] != self.M:
            raise ValueError(
                f"x_block has {x_block.shape[1]} channels but filter expects M={self.M}."
            )

        y_block = np.zeros(T, dtype=float)
        e_block = np.zeros(T, dtype=float)

        for n in range(T):
            # Shift each channel's N-tap delay line independently, then
            # write the new sample into position 0 of that channel's segment.
            #   x_buf = [ch0: x(n) x(n-1)...x(n-N+1) | ch1: ... | ...]
            for m in range(self.M):
                start = m * self.N
                self.x_buf[start + 1 : start + self.N] = \
                    self.x_buf[start : start + self.N - 1]
                self.x_buf[start] = x_block[n, m]

            self._compute_step(float(d_block[n]))

            y_block[n] = self.y
            e_block[n] = self.e

        return y_block, e_block
    

# =============================================================================
# Multichannel LMS
# =============================================================================

class MultiChannelLMS(MultiFilter):
    """
    Multichannel LMS filter.

    Maintains one N-tap FIR per reference channel. All tap banks are updated
    jointly from the single scalar error e[n] = d[n] - y[n], where y[n] is
    the sum of all M channel outputs. This is equivalent to a single LMS filter
    operating on the NM-length supervector x_buf.

    Parameters
    ----------
    N  : taps per channel
    M  : number of reference channels
    mu : LMS step size (shared across all channels)
         Rule of thumb: mu < 1 / (M * N * max_input_power)
    """

    def __init__(self, N: int, M: int, mu: float = 0.01):
        super().__init__(N, M)
        self.mu = float(mu)

    def _compute_step(self, d_n: float):
        # y[n] = w^T * x_buf  (dot over the full NM supervector)
        self.y = float(self.taps @ self.x_buf)
        self.e = d_n - self.y
        # Standard LMS update — identical in form to MonoLMS
        self.taps += self.mu * self.e * self.x_buf


# =============================================================================
# Multichannel NLMS
# =============================================================================

class MultiChannelNLMS(IIRPostFilter, MultiFilter):
    """
    Multichannel NLMS filter.

    Normalizes the step size by the total power across all M delay buffers:
        mu_eff = mu / (eps + ||x_buf||^2)
    where x_buf is the full NM supervector. This keeps the effective step size
    scale-invariant regardless of how many channels or taps are used.

    Parameters
    ----------
    N          : taps per channel
    M          : number of reference channels
    mu         : initial step size (typically 0.1–1.0 for NLMS)
    eps        : regularization — prevents divide-by-zero on silent input
    varying_mu : exponentially decay mu toward mu_floor each sample
    mu_floor   : lower bound on mu when varying_mu=True
    factor     : per-sample decay multiplier (e.g. 0.9999)
    """

    def __init__(
        self,
        N: int,
        M: int,
        mu: float = 0.5,
        eps: float = 1e-7,
        *,
        varying_mu: bool = False,
        mu_floor: float = 1e-5,
        factor: float = 0.9999,
        iir_enable: bool = False,
        iir_on: str = "e",
        taps_path: str = "",
    ):
        MultiFilter.__init__(self, N, M)
        self.mu         = float(mu)
        self.eps        = float(eps)
        self.varying_mu = bool(varying_mu)
        self.mu_floor   = float(mu_floor)
        self.factor     = float(factor)
        self._init_iir(iir_enable, iir_on, taps_path)

    def _compute_step(self, d_n: float):
        self.y = float(self.taps @ self.x_buf)
        self.e = d_n - self.y

        # Normalize by total power of the full NM supervector
        denom = self.eps + float(self.x_buf @ self.x_buf)
        self.taps += (self.mu / denom) * self.e * self.x_buf

        if self.varying_mu:
            self.mu = max(self.mu * self.factor, self.mu_floor)

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        y_block, e_block = super().process_block(x_block, d_block)
        return self._apply_iir(y_block, e_block)


# =============================================================================
# Multichannel RLS
# =============================================================================

class MultiChannelRLS(IIRPostFilter, MultiFilter):
    """
    Multichannel RLS filter.

    The inverse correlation matrix P is now (NM x NM), capturing the full
    cross-channel and cross-lag covariance structure. This allows the filter to
    exploit correlations between reference channels and between tap delays
    simultaneously — the key advantage over running M independent mono filters.

    The per-sample update is identical in form to MonoRLS:
        Px  = P @ x_buf                       (NM vector)
        k   = Px / (lam + x_buf^T @ Px)       (NM Kalman gain)
        y   = taps^T @ x_buf                  (scalar)
        e   = d - y                            (scalar)
        w  += k * e                            (NM tap update)
        P   = (1/lam) * (P - outer(k, Px))    (NM x NM rank-1 downdate)

    Memory note: P is (NM)^2 floats.
        N=64,  M=8  -> 512^2  * 8 B =   2 MB  (fast)
        N=128, M=8  -> 1024^2 * 8 B =   8 MB  (fine)
        N=256, M=8  -> 2048^2 * 8 B =  32 MB  (slow per sample — reduce N or M)
    Recommended: keep N*M <= 512 for interactive use.

    Parameters
    ----------
    N     : taps per channel
    M     : number of reference channels
    lam   : forgetting factor (0.999–0.99999 typical)
    delta : initial P = (1/delta)*I  — larger = faster early adaptation
    """

    def __init__(
        self,
        N: int,
        M: int,
        lam: float = 0.999,
        delta: float = 1e-2,
        *,
        iir_enable: bool = False,
        iir_on: str = "e",
        taps_path: str = "",
    ):
        MultiFilter.__init__(self, N, M)
        self.lam     = float(lam)
        self.lam_inv = 1.0 / self.lam
        # P is (NM x NM) — the full cross-channel inverse correlation matrix
        self.P = (1.0 / delta) * np.eye(self.NM, dtype=float)
        self._init_iir(iir_enable, iir_on, taps_path)

    def _compute_step(self, d_n: float):
        # Kalman gain (NM vector)
        Px    = self.P @ self.x_buf
        denom = self.lam + float(self.x_buf @ Px)
        k     = Px / denom

        # Output and error
        self.y = float(self.taps @ self.x_buf)
        self.e = d_n - self.y

        # Tap and P updates — structurally identical to MonoRLS
        self.taps += k * self.e
        self.P     = self.lam_inv * (self.P - np.outer(k, Px))

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        y_block, e_block = super().process_block(x_block, d_block)
        return self._apply_iir(y_block, e_block)


# =============================================================================
# PCA Filter
# =============================================================================

class PCA_Filter:
    """
    Robust batch PCA-based fetal ECG extractor.

    Operates on a multichannel abdominal block A of shape (M, T):
      - M channels (rows), T time samples (columns)
      - Mean-centers each channel across time before SVD
      - Suppresses the top-k PCs (dominant maternal components)
      - Returns the residual as the fetal estimate

    Three robustness issues are handled explicitly:

    1. Sign ambiguity
       SVD eigenvectors are sign-arbitrary. The fetal residual may emerge
       inverted relative to the true fetal ECG. Correct this by aligning
       the sign of each residual PC row (Vh[k:, :]) so that its dominant
       deflection (max absolute value) is positive before reconstructing.
       Additionally, after collapsing channels we compare the skewness of
       the result to its negation: fetal QRS complexes produce a positive
       skew when oriented correctly, so we flip if skew < 0.

    2. DC offset
       After zeroing the top-k singular values and adding the channel mean
       back, a residual DC offset accumulates from the mean reintroduction.
       We explicitly mean-subtract e_mean after reconstruction.

    3. Scale collapse from channel averaging
       Averaging M residual channels attenuates the fetal signal by ~1/sqrt(M)
       relative to noise. Instead of a plain mean we use a variance-weighted
       combination: channels with higher residual variance contribute more,
       which preferentially weights channels where fetal signal is stronger.

    Parameters
    ----------
    k               : number of PCs to suppress (maternal rank; typically 1–2)
    num_a_channels  : expected number of abdominal input channels M

    Notes
    -----
    - PCA is a batch method: it sees the full block at once. Unlike LMS/RLS,
      there are no per-sample tap updates.
    - Minimum recommended block size: ~2 cardiac cycles (~1 s at 1 kHz).
      Shorter blocks produce unreliable covariance estimates.
    - Set k < num_a_channels strictly. k=1 removes one dominant maternal PC;
      k=2 is useful when respiration or twin gestation raises maternal rank.
    """

    def __init__(self, k: int, num_a_channels: int):
        self.k = int(k)
        self.num_a_channels = int(num_a_channels)

    @staticmethod
    def _fix_pc_signs(Vh: np.ndarray) -> np.ndarray:
        """
        Enforce a deterministic sign convention on PC rows.

        For each row of Vh (each PC time course), flip the sign so that
        the sample with the largest absolute value is positive. This removes
        the SVD sign ambiguity before reconstruction.

        Parameters
        ----------
        Vh : (r, T) — right singular vectors (PC time courses as rows)

        Returns
        -------
        Vh_fixed : (r, T) — sign-corrected copy
        """
        Vh_fixed = Vh.copy()
        for i in range(Vh_fixed.shape[0]):
            row = Vh_fixed[i]
            if row[np.argmax(np.abs(row))] < 0:
                Vh_fixed[i] = -row
        return Vh_fixed

    @staticmethod
    def _variance_weighted_mean(A_fetal: np.ndarray) -> np.ndarray:
        """
        Collapse (M, T) residual matrix to (T,) using variance weights.

        Channels with higher temporal variance carry more fetal signal energy
        and are upweighted. This outperforms a plain mean when channel SNRs
        differ significantly.

        Parameters
        ----------
        A_fetal : (M, T) residual matrix (mean-subtracted per channel)

        Returns
        -------
        e : (T,) weighted combination
        """
        variances = np.var(A_fetal, axis=1)        # (M,)
        total_var = variances.sum()
        if total_var < 1e-30:
            # Degenerate: all channels flat — fall back to plain mean
            return np.mean(A_fetal, axis=0)
        weights = variances / total_var            # (M,) sums to 1
        return weights @ A_fetal                   # (T,)

    @staticmethod
    def _correct_sign_by_skewness(e: np.ndarray) -> np.ndarray:
        """
        Flip e if its skewness is negative.

        Fetal QRS complexes, when correctly oriented, produce sharp positive
        deflections (R-peaks) surrounded by smaller negative deflections,
        giving the signal a positive skewness. If skewness < 0, the signal
        is inverted and we flip it.

        This is a soft heuristic that works well for typical fECG morphology.
        It may not hold for unusual lead orientations or foetal presentations
        where R-peaks are intrinsically negative — in those cases set
        auto_sign=False in process_block and flip manually after the call.

        Parameters
        ----------
        e : (T,) fetal estimate

        Returns
        -------
        e or -e depending on skewness sign
        """
        from scipy.stats import skew
        return e if skew(e) >= 0 else -e

    def _pca_decompose(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose (M, T) abdominal matrix into maternal estimate and fetal residual.

        Steps
        -----
        1. Mean-center each channel (row) across time.
        2. Economy SVD → U (M,M), S (M,), Vh (M,T).
        3. Fix sign of residual PC rows in Vh to remove SVD sign ambiguity.
        4. Reconstruct maternal (top-k PCs) and fetal (remaining PCs) matrices.
        5. Collapse fetal matrix to scalar with variance-weighted channel mean.
        6. Remove DC from both outputs.

        Parameters
        ----------
        A : (M, T) float array

        Returns
        -------
        y_mean : (T,) maternal estimate
        e_mean : (T,) fetal estimate (DC-free, sign-corrected before skewness check)
        """
        M, T = A.shape

        # Step 1 — mean-center
        mu = np.mean(A, axis=1, keepdims=True)     # (M, 1)
        A_centered = A - mu                         # (M, T)

        # Step 2 — economy SVD
        U, S, Vh = np.linalg.svd(A_centered, full_matrices=False)
        # U: (M, M), S: (M,), Vh: (M, T)

        # Step 3 — fix signs of residual PCs (indices k..M-1) only
        # Maternal PCs (0..k-1) sign doesn't matter for the fetal output.
        Vh_fixed = Vh.copy()
        Vh_fixed[self.k:] = self._fix_pc_signs(Vh[self.k:])

        # Step 4 — reconstruct maternal (rank-k) and fetal (rank-(M-k))
        # Maternal: keep only top-k singular values
        S_maternal        = np.zeros_like(S)
        S_maternal[:self.k] = S[:self.k]
        A_maternal = U @ np.diag(S_maternal) @ Vh_fixed   # (M, T)
        A_maternal += mu

        # Fetal: zero the top-k singular values, keep the rest
        S_fetal        = S.copy()
        S_fetal[:self.k] = 0.0
        A_fetal = U @ np.diag(S_fetal) @ Vh_fixed         # (M, T)
        # Do NOT add mu back to fetal — the channel means are dominated by
        # maternal DC and noise; re-adding them pollutes the residual.
        # We subtract the per-channel mean of the residual instead.
        A_fetal -= np.mean(A_fetal, axis=1, keepdims=True)

        # Step 5 — collapse channels with variance weighting
        y_mean = np.mean(A_maternal, axis=0)               # (T,) plain mean is fine for y
        e_mean = self._variance_weighted_mean(A_fetal)     # (T,) weighted

        # Step 6 — remove any residual DC
        y_mean -= np.mean(y_mean)
        e_mean -= np.mean(e_mean)

        return y_mean, e_mean

    def process_block(
        self,
        x_block: np.ndarray,
        d_block: np.ndarray,     # accepted for API compatibility; ignored by PCA
        *,
        auto_sign: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run PCA on an abdominal block and return (y, e).

        Parameters
        ----------
        x_block   : (T, M) float array — T samples, M abdominal channels.
                    Pass the ABDOMINAL channels here, not a thoracic reference.
        d_block   : (T,) float array — ignored; present for API compatibility.
        auto_sign : if True (default), apply skewness-based sign correction to e.
                    Set False if your fetal R-peaks are known to be negative, or
                    if you want to handle sign manually after the call.

        Returns
        -------
        y_block : (T,) maternal estimate
        e_block : (T,) fetal estimate — DC-free, sign-corrected
        """
        x_block = np.asarray(x_block, dtype=float)
        if x_block.ndim == 1:
            x_block = x_block[:, np.newaxis]

        T, M = x_block.shape

        if M != self.num_a_channels:
            raise ValueError(
                f"x_block has {M} channels but PCA_Filter expects "
                f"num_a_channels={self.num_a_channels}."
            )
        if self.k >= M:
            raise ValueError(
                f"k={self.k} must be strictly less than num_a_channels={M}. "
                f"Suppressing all components leaves no fetal signal."
            )
        if T < 2 * M:
            raise ValueError(
                f"Block too short: T={T} samples, M={M} channels. "
                f"Need T >= 2*M for a reliable covariance estimate. "
                f"Increase block_size or reduce num_a_channels."
            )

        # Transpose to (M, T): channels as rows, time as columns
        A = x_block.T

        y_block, e_block = self._pca_decompose(A)

        # Skewness-based global sign correction on fetal estimate
        if auto_sign:
            e_block = self._correct_sign_by_skewness(e_block)

        return y_block, e_block
