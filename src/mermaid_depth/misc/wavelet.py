from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from obspy import read

# -------------------------
# MERMAID wavelet code (integer lifting)
# -------------------------
def _floor_plus_half(z):
    return np.floor(z + 0.5).astype(np.int64)

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def clip_mallat(coeffs, ncasc=5, clip_by_level=None, clip_A=None):
    """
    Clip in Mallat-ordered coefficient vector:
        w = [A_ncasc | D_ncasc | ... | D_1]
    clip_A: int or None
    clip_by_level: dict {j: clip} where j=1..ncasc and clip is:
        - None -> do not clip that band
        - 0    -> zero that band
        - >0   -> clip to [-clip, +clip]
    """
    c = coeffs.copy()
    N = len(c)

    # ---- Clip A_ncasc (approximation) ----
    if clip_A is not None:
        a_len = N // (2**ncasc)
        c[:a_len] = np.clip(c[:a_len], -clip_A, clip_A)

    # ---- Clip detail bands ----
    pos = N // (2**ncasc)
    for j in range(ncasc, 0, -1):
        dlen = N // (2**j)
        if clip_by_level and j in clip_by_level:
            clip = clip_by_level[j]
            if clip is not None:
                c[pos:pos+dlen] = np.clip(c[pos:pos+dlen], -clip, clip)
        pos += dlen

    return c

def fmermaid_clip(x, clip_detail=None, clip_approx=None, ncasc=5):
    x = x.copy()
    lx = len(x)

    for level in range(1, ncasc + 1):
        # PREDICT
        for i in range(1, lx - 1, 2):
            pred = _floor_plus_half((x[i - 1] + x[i + 1]) / 2)
            x[i] -= pred

        # CLIP DETAIL (odd indices)
        if clip_detail is not None:
            x[1:lx:2] = np.clip(x[1:lx:2], -clip_detail, clip_detail)

        # UPDATE
        for i in range(4, lx - 3, 2):
            upd = _floor_plus_half(
                (-3*x[i-3] + 19*x[i-1] + 19*x[i+1] - 3*x[i+3]) / 64
            )
            x[i] += upd

        # CLIP APPROX (even indices)
        if clip_approx is not None:
            x[0:lx:2] = np.clip(x[0:lx:2], -clip_approx, clip_approx)

        # REARRANGE (Mallat order)
        x = np.concatenate([x[0:lx:2], x[1:lx:2], x[lx:]])
        lx //= 2

    return x

def imermaid(x, ncasc: int = 5):
    """
    Inverse MERMAID transform corresponding to fmermaid_clip (integer lifting).
    Input coefficients must be integer and length must be power-of-two.
    """
    x = np.asarray(x).ravel()

    if not _is_power_of_two(len(x)):
        raise ValueError("Input array must have length a power of two")

    if not np.all(np.round(x) == x):
        raise ValueError("Input coefficients must be integer-valued")

    x = x.astype(np.int64, copy=True)
    N = len(x)

    for level in range(ncasc, 0, -1):
        lx = N // (2 ** level)

        # inverse UPDATE
        for n in range(3, lx):
            i = n - 1
            upd = _floor_plus_half(
                (-3 * x[lx + n - 3] + 19 * x[lx + n - 2] + 19 * x[lx + n - 1] - 3 * x[lx + n]) / 64.0
            )
            x[i] = x[i] - upd

        # inverse PREDICT
        for n in range(1, lx):
            pred = _floor_plus_half((x[n - 1] + x[n]) / 2.0)
            x[lx + n - 1] = x[lx + n - 1] + pred

        # undo REARRANGE
        y0 = x[0:lx].copy()
        y1 = x[lx:2*lx].copy()
        x[0:2*lx:2] = y0
        x[1:2*lx:2] = y1

    return x

# -------------------------
# Helpers: rounding + power-of-two handling
# -------------------------
def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

def sac_to_int64_round(x_float: np.ndarray, mode: str = "rint") -> np.ndarray:
    """
    Convert SAC float trace to int64 for integer lifting.
    Assumes MERMAID records already have compact amplitude (e.g. ~[-100,100]).

    mode:
      - "rint": np.rint (banker's rounding)
      - "half_away": round half away from zero (more symmetric for negatives)
    """
    x = np.asarray(x_float, dtype=float)

    if mode == "rint":
        return np.rint(x).astype(np.int64)

    if mode == "half_away":
        y = np.sign(x) * np.floor(np.abs(x) + 0.5)
        return y.astype(np.int64)

    raise ValueError("mode must be 'rint' or 'half_away'")

def int64_to_float(x_int: np.ndarray) -> np.ndarray:
    return np.asarray(x_int, dtype=np.int64).astype(float)

def pad_or_trim_power_of_two(x: np.ndarray, mode: str = "pad"):
    x = np.asarray(x)
    n0 = len(x)

    if _is_power_of_two(n0):
        return x, n0

    if mode == "pad":
        n1 = next_pow2(n0)
        out = np.zeros(n1, dtype=x.dtype)
        out[:n0] = x
        return out, n0

    if mode == "trim":
        n1 = 2 ** int(np.floor(np.log2(n0)))
        return x[:n1], n0

    raise ValueError("mode must be 'pad' or 'trim'")

# -------------------------
# Plot annotation helpers: show what was clipped
# -------------------------
def _format_clip_summary(ncasc: int, clip_by_level: dict[int, int] | None, clip_A):
    """
    Human-readable summary of which Mallat bands were modified.
    w = [A_ncasc | D_ncasc | ... | D_1]
    """
    parts = []

    # Approx band
    if clip_A is None:
        parts.append(f"A{ncasc}: kept")
    else:
        if clip_A == 0:
            parts.append(f"A{ncasc}: zeroed")
        else:
            parts.append(f"A{ncasc}: clipped ±{clip_A}")

    # Details
    if clip_by_level is None:
        parts.append(f"D{ncasc}..D1: kept")
        return " | ".join(parts)

    d_parts = []
    for j in range(ncasc, 0, -1):
        if j in clip_by_level:
            v = clip_by_level[j]
            if v is None:
                d_parts.append(f"D{j}: kept")
            elif v == 0:
                d_parts.append(f"D{j}: zeroed")
            else:
                d_parts.append(f"D{j}: clipped ±{v}")
        else:
            d_parts.append(f"D{j}: kept")

    parts.append(", ".join(d_parts))
    return " | ".join(parts)

def _bands_zeroed(ncasc: int, clip_by_level: dict[int, int] | None, clip_A):
    z = []
    if clip_A == 0:
        z.append(f"A{ncasc}")
    if clip_by_level:
        for j in range(1, ncasc + 1):
            if j in clip_by_level and clip_by_level[j] == 0:
                z.append(f"D{j}")
    return z

# -------------------------
# Main: read SAC, transform, plot (and save pdf)
# -------------------------
def plot_mermaid_on_sac(
    sac_path: str | Path,
    ncasc: int = 5,
    pow2_mode: str = "pad",                 # "pad" or "trim"
    clip_by_level: dict[int, int] | None = None,
    clip_A=None,
    clip_detail=None,
    clip_approx=None,
    rounding_mode: str = "rint",            # "rint" or "half_away"
    save_path=None,
):
    sac_path = Path(sac_path)
    tr = read(str(sac_path))[0]

    # Original float trace
    x0 = tr.data.astype(float)

    # Time axis
    dt = float(tr.stats.delta)
    t0 = float(getattr(tr.stats.sac, "b", 0.0)) if hasattr(tr.stats, "sac") else 0.0
    t = t0 + np.arange(len(x0)) * dt

    # Convert to int64 by rounding only (no explicit scale/normalization)
    x_int = sac_to_int64_round(x0, mode=rounding_mode)

    # Ensure power-of-two length
    x_pow2, n_orig = pad_or_trim_power_of_two(x_int, mode=pow2_mode)

    if len(x_pow2) < (2 ** ncasc):
        raise ValueError(f"Signal too short for ncasc={ncasc}. Need length >= {2**ncasc}.")

    # Forward transform
    w = fmermaid_clip(x_pow2, clip_detail=clip_detail, clip_approx=clip_approx, ncasc=ncasc)

    # Decide whether to clip at all (avoid truthy dict with only None values)
    do_clip = (clip_A is not None) or (
        clip_by_level is not None and any(v is not None for v in clip_by_level.values())
    )

    # Optional Mallat clipping (details and/or A_ncasc)
    w_used = clip_mallat(w, ncasc=ncasc, clip_by_level=clip_by_level, clip_A=clip_A) if do_clip else w

    # Inverse transform
    x_rec_int = imermaid(w_used, ncasc=ncasc)

    # Undo padding/trim to match original plotting length
    if pow2_mode == "pad":
        x_rec_int = x_rec_int[:n_orig]
    elif pow2_mode == "trim":
        x0 = x0[:len(x_rec_int)]
        t = t[:len(x_rec_int)]

    # Back to float for plotting (integer-valued waveform)
    x1 = int64_to_float(x_rec_int)

    # Build plot with "what was clipped" annotation
    fig = plt.figure(figsize=(12, 4))
    plt.plot(t, x0, label="Original (float)", linewidth=1.0)
    plt.plot(t, x1, label="Modified (MERMAID, int)", linewidth=1.0, linestyle="--", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    clip_summary = _format_clip_summary(ncasc, clip_by_level, clip_A)
    zeroed = _bands_zeroed(ncasc, clip_by_level, clip_A)

    title = f"{sac_path.name} | ncasc={ncasc}"
    if zeroed:
        title += f" | zeroed: {', '.join(zeroed)}"
    plt.title(title)

    ax = plt.gca()
    ax.text(
        0.01, 0.99,
        clip_summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.5"),
    )

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)

    return x0, x1, t, w_used

# -------------------------
# Batch runner: ./Q01/*.sac -> ./plots/*.pdf
# -------------------------
if __name__ == "__main__":
    input_dir = Path("./Mermaid_Data_Joel")
    output_dir = Path("./plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    sac_files = sorted(input_dir.glob("*.sac"))
    print(f"Found {len(sac_files)} SAC files")

    # Example: zero out finest 4 bands, keep only D5 + A5 (if clip_A is None)
    # Use None for "do not clip that level"
    clip_levels = {1: 0, 2: 0, 3: 0, 4: 0, 5: None}
    clip_A = 80

    for sac_file in sac_files[:0]:
        out_pdf = output_dir / f"{sac_file.stem}.pdf"
        print(f"Processing {sac_file.name} -> {out_pdf.name}")

        try:
            plot_mermaid_on_sac(
                sac_path=sac_file,
                ncasc=5,
                pow2_mode="trim",
                clip_by_level=clip_levels,
                clip_A=clip_A,
                clip_detail=None,
                clip_approx=None,
                rounding_mode="half_away",  # "rint" or "half_away"
                save_path=out_pdf,
            )
        except Exception as e:
            print(f"Failed on {sac_file.name}: {e}")
