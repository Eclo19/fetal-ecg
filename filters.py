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

