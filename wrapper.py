import numpy as np
import wfdb
from pathlib import Path


def _load_wfdb_record_matrix(record_dir: Path, record_name: str, num_samples: int = 7500):
    """
    Load a WFDB record and return:
      X: (N, n_sig) float matrix (physical units)
      fs: sampling rate
    """
    record_path = str(record_dir / record_name)  # full path without extension
    rec = wfdb.rdrecord(record_path, physical=True)

    X = rec.p_signal  # (N_total, n_sig)
    fs = float(rec.fs)

    N_total = X.shape[0]
    N = min(int(num_samples), N_total)

    return X[:N, :].astype(float, copy=True), fs


class Wrapper:
    def __init__(self, partial_path: str):
        """
        partial_path examples:
          physionet.org/files/fecgsyndb/1.0.0/sub01/snr12dB/sub01_snr12dB_l1_c0
          physionet.org/files/fecgsyndb/1.0.0/sub01/snr12dB/sub01_snr12dB_l1_c5
          physionet.org/files/fecgsyndb/1.0.0/sub01/snr12dB/sub01_snr12dB_l1   (baseline style)
        """
        self.partial_path = partial_path

        # Split at version folder
        left, right = partial_path.split("1.0.0/", 1)
        self.all_data_dir = left + "1.0.0/"  # ends with 1.0.0/
        parts = right.split("/")

        # Get basic dirs
        self.sub = parts[0]
        self.snr = parts[1]
        self.record_base = parts[2]

        # Case token (c0..c5) if present
        self.c = None
        tokens = self.record_base.split("_")
        for tok in tokens:
            if tok.startswith("c") and tok[1:].isdigit():
                print("ADDING CASE.")
                self.c = tok
                break

    def get_mixed_ecg(self, x_ch=None, d_ch=None, num_samples: int = 7500, noise=True):
        """
        Build a mixed 34-channel measurement by summing available component records
        in the directory corresponding to self.sub/self.snr.

        Intended usage:
          w = Wrapper(".../sub01/snr12dB/sub01_snr12dB_l1_c0")
          Xmix, fs = w.get_mixed_ecg(num_samples=7500)

        Returns:
          Xmix: (N, 34) ndarray
          fs: float
        """
        # Directory where the WFDB records live
        record_dir = Path(self.all_data_dir) / self.sub / self.snr
        if not record_dir.exists():
            raise FileNotFoundError(f"Record directory not found: {record_dir}")

        # Base prefix like: sub01_snr12dB_l1_c0   (or without _cZ)
        base = self.record_base

        # Helper: does a .hea exist for a given record_name?
        def has_record(name: str) -> bool:
            return (record_dir / f"{name}.hea").exists()

        # --- Decide what components to include ---
        # Always try: mecg + all fecgN present + all noise* present.
        # Note: This constructs mixtures even if the dataset doesn’t provide an explicit "aecg" record.
        component_names = []

        # maternal
        mecg_name = f"{base}_mecg"
        if has_record(mecg_name):
            component_names.append(mecg_name)

        # fetal: fecg1..fecg5 (include those that exist)
        for k in range(1, 6):
            fecg_name = f"{base}_fecg{k}"
            if has_record(fecg_name):
                component_names.append(fecg_name)

        if noise:
            # noise: sometimes noise, noise1, noise2, ...; include whatever exists
            # Try common patterns first.
            print("ADDING NOISE")
            for nm in ("noise", "noise1", "noise2", "noise3", "noise4", "noise5"):
                noise_name = f"{base}_{nm}"
                if has_record(noise_name):
                    component_names.append(noise_name)

        if not component_names:
            raise FileNotFoundError(
                f"No component records found for base='{base}' in {record_dir}.\n"
                f"Expected files like '{base}_mecg.hea', '{base}_fecg1.hea', '{base}_noise*.hea'."
            )

        # --- Load and sum as 34-channel matrices ---
        Xsum = None
        fs_out = None

        for name in component_names:
            print(f"Adding component {name}")
            X, fs = _load_wfdb_record_matrix(record_dir, name, num_samples=num_samples)

            if fs_out is None:
                fs_out = fs
            elif abs(fs - fs_out) > 1e-9:
                raise ValueError(f"Sampling rate mismatch: {name} has fs={fs}, expected fs={fs_out}")

            # Ensure 34 channels (the database standard). If not, still sum by min channels.
            if Xsum is None:
                Xsum = X
            else:
                # Trim to common length and common channel count 
                N = min(Xsum.shape[0], X.shape[0])
                C = min(Xsum.shape[1], X.shape[1])
                Xsum = Xsum[:N, :C] + X[:N, :C]

        # Allow for specific x and d channel extraction 
        if x_ch is not None or d_ch is not None:
            C = Xsum.shape[1]
            if x_ch is not None and (x_ch < 0 or x_ch >= C):
                raise ValueError(f"x_ch={x_ch} out of range (0..{C-1})")
            if d_ch is not None and (d_ch < 0 or d_ch >= C):
                raise ValueError(f"d_ch={d_ch} out of range (0..{C-1})")

            x = Xsum[:, x_ch].copy() if x_ch is not None else None
            d = Xsum[:, d_ch].copy() if d_ch is not None else None
            return Xsum, fs_out, x, d

        return Xsum, fs_out

    def get_pure_channel(self, name: str, ch: int, num_samples: int = 7500, *, return_fs: bool = False):
        """
        Load a 'pure' WFDB record associated with this wrapper's base record, and return one channel.

        name: suffix appended to record_base, e.g. "mecg", "fecg1", "fecg2", etc.
            (Do NOT pass a full record name that already includes record_base.)
        ch: channel index to extract
        """
        if name is None or not isinstance(name, str) or name.strip() == "":
            raise ValueError("name must be a non-empty suffix string like 'mecg' or 'fecg1'.")

        if ch is None:
            raise ValueError("ch must be an integer channel index.")

        record_dir = Path(self.all_data_dir) / self.sub / self.snr
        if not record_dir.exists():
            raise FileNotFoundError(f"Record directory not found: {record_dir}")

        # Prevent accidental double-prefixing
        if name.startswith(self.record_base):
            full_name = name
        else:
            full_name = f"{self.record_base}_{name}"

        X, fs = _load_wfdb_record_matrix(record_dir, full_name, num_samples=num_samples)

        if not (0 <= ch < X.shape[1]):
            raise IndexError(f"Channel {ch} out of range for {full_name}: X has {X.shape[1]} channels.")

        out = X[:, ch]
        return (out, fs) if return_fs else out


# Example
if False:
    myWrapper = Wrapper(
        partial_path="/Users/ericoliviera/Desktop/ECE NU/ELECT_ENG_495 - Cardiovascular Instrumentation/Project/physionet.org/files/fecgsyndb/1.0.0/sub01/snr12dB/sub01_snr12dB_l1_c0"
    )
    Xmix, fs = myWrapper.get_mixed_ecg(num_samples=7500, noise=False)
    print(f"Xmin shape = {Xmix.shape}")