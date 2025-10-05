# SpecFall — Waterfall plotting for radio-astronomy MS Datasets
# Copyright (C) 2025  Prabhanjan H. Kulkarni <astro.ptabhanjan@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import datetime as _dt
import matplotlib.dates as mdates
from ..utils.logging import log
import os

POL_ALIASES = {"xx": 0, "yy": 1, "rr": 0, "ll": 1, "xy": 0, "yx": 1}

def _ms_time_to_mpl_dates(times_sec: np.ndarray) -> np.ndarray:
    """
    Convert MS TIME (seconds since MJD 0; i.e., MJD reference 1858-11-17) to Matplotlib date numbers.
    Returns array of floats suitable for imshow extent and DateFormatter.
    """
    # MJD epoch
    epoch = _dt.datetime(1858, 11, 17, 0, 0, 0)
    # Vectorized conversion
    return mdates.date2num([epoch + _dt.timedelta(seconds=float(t)) for t in times_sec])

def _nan_group_mean_by_time(values: np.ndarray, times_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate along axis-0 by identical times in `times_1d`, computing nanmean.
    values: (N, C, P)  float array with NaNs for flagged samples
    times_1d: (N,)  numeric times aligned to values

    Returns:
      unique_times: (G,) sorted unique time values
      mean_vals:    (G, C, P) nan-mean per time-bin
    """
    if values.size == 0:
        return np.array([], dtype=times_1d.dtype), values[:0]
    order = np.argsort(times_1d)
    t_sorted = times_1d[order]
    v_sorted = values[order]

    # Group boundaries
    new_group = np.empty_like(t_sorted, dtype=bool)
    new_group[0] = True
    np.not_equal(t_sorted[1:], t_sorted[:-1], out=new_group[1:])
    idx_start = np.flatnonzero(new_group)
    idx_end = np.r_[idx_start[1:], t_sorted.size]

    # Valid mask & sums/counts with NaN handling
    valid = ~np.isnan(v_sorted)
    v_filled = np.where(valid, v_sorted, 0.0)

    sums = np.add.reduceat(v_filled, idx_start, axis=0)
    cnts = np.add.reduceat(valid.astype(np.float64), idx_start, axis=0)

    # Avoid divide-by-zero; keep NaN where no valid samples
    out = np.divide(sums, cnts, out=np.full_like(sums, np.nan), where=cnts > 0)

    unique_times = t_sorted[idx_start]
    return unique_times, out

class WaterfallPlotter:
    def __init__(self, ms):
        self.ms = ms

    def waterfall(
        self,
        x_axis: str = "freq",          # "freq" or "channel"
        log_amp: bool = True,
        pol: str | int | None = None,  # override selection
        layout: str = "tb",
        vmax: float | None = None,
        vmin: float | None = None,
        cmap: str = "viridis",
        title: str | None = None,
        amp_scale: float = 1.0,
        amp_unit: str = "Jy",
        mhz_tick: float | None = 1.0,
        mhz_label: float | None = 5.0,
        outdir: str | None = None,
        outfile: str | None = None,
        baseline: str | tuple[int, int] | list[tuple[int, int]] = "avg",
        bl_cols: int = 2,
    ):
        """
        Plot baseline-wise waterfalls, one PNG per (polarisation, baseline).
        This simpler mode mirrors the reference script: for each selected scan
        we read all rows via TAQL, compute per-row amplitudes (|DATA| masked by FLAG),
        bucket by (pol, baseline), sort by time and plot a waterfall with Y as
        time index (tick labels show UTC).
        """
        from casacore.tables import table
        from collections import defaultdict
        import matplotlib.dates as _mdates
        from datetime import datetime, timezone

        meta = self.ms.meta
        sel = self.ms._sel
        pol = pol if pol is not None else sel.pol

        # Channel window selection
        c0, c1 = _resolve_channel_window(meta, sel)
        nchan_sel = c1 - c0

        # Determine scans
        scans = sel.scan
        if scans is None:
            scans = tuple(np.unique(meta.scans).tolist())
        else:
            scans = tuple(scans)

        # Prepare frequency / channel axis
        if x_axis == "channel":
            x_min, x_max = 0, nchan_sel - 1
            xlabel = "Channel"
        else:
            freqs_mhz = (self.ms.meta.chan_freq[c0:c1] / 1e6)
            x_min, x_max = float(freqs_mhz[0]), float(freqs_mhz[-1])
            xlabel = "Frequency [MHz]"

        # Output directory
        if outdir is None:
            outdir = "."
        os.makedirs(outdir, exist_ok=True)

        # Data buckets: dict[pol_index]["a-b"] -> list of (time_sec, amp_vec)
        baseline_data: dict[int, dict[str, list[tuple[float, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))

        with table(self.ms.path, readonly=True) as T:
            # Iterate requested scans using TAQL to handle non-contiguous rows
            for sc in scans:
                q = T.query(f"SCAN_NUMBER=={int(sc)}")
                try:
                    nrows = q.nrows()
                    if nrows == 0:
                        log.warning(f"No rows for scan {sc}; skipping.")
                        continue

                    data = q.getcol(meta.data_col)         # (rows, nchan, npol) or (rows, npol, nchan)
                    flag = q.getcol("FLAG")                # same shape
                    a1   = q.getcol("ANTENNA1")            # (rows,)
                    a2   = q.getcol("ANTENNA2")            # (rows,)
                    times = q.getcol("TIME")               # (rows,)
                finally:
                    q.close()

                # Normalize axes to (rows, nchan, npol)
                if data.ndim != 3 or flag.ndim != 3:
                    raise ValueError(f"Unexpected DATA/FLAG dimensionality: DATA {data.shape}, FLAG {flag.shape}")
                expected_nchan = int(meta.nchan)
                if data.shape[1] == expected_nchan:
                    pass
                elif data.shape[2] == expected_nchan:
                    data = np.transpose(data, (0, 2, 1))
                    flag = np.transpose(flag, (0, 2, 1))
                else:
                    if flag.shape[1] == expected_nchan:
                        pass
                    elif flag.shape[2] == expected_nchan:
                        data = np.transpose(data, (0, 2, 1))
                        flag = np.transpose(flag, (0, 2, 1))
                    else:
                        raise ValueError(
                            f"Cannot locate channel axis: expected {expected_nchan} in {data.shape} / {flag.shape}")

                # Channel cut
                data = data[:, c0:c1, :]
                flag = flag[:, c0:c1, :]

                # Amplitude masked by flags
                amp = np.abs(data)
                amp = np.where(~flag, amp, np.nan)  # (rows, nchan_sel, npol)
                # Convert amplitude to Jansky scale (1 Jy = 1e-26 W/m^2/Hz)
                amp *= 1e26

                # Bucket per (pol, baseline)
                npol_here = amp.shape[-1]
                for r in range(amp.shape[0]):
                    bl_id = f"{min(int(a1[r]), int(a2[r]))}-{max(int(a1[r]), int(a2[r]))}"
                    t = float(times[r])
                    for p_idx in range(npol_here):
                        # Selection of pol: if user asked a concrete pol index/string, filter here
                        if isinstance(pol, int) and p_idx != pol:
                            continue
                        if isinstance(pol, str) and pol.lower() != "both":
                            # Map alias to index, skip others
                            p_alias = POL_ALIASES.get(pol.lower(), None)
                            if p_alias is not None and p_idx != int(p_alias):
                                continue
                        arow = amp[r, :, p_idx]
                        baseline_data[p_idx][bl_id].append((t, arow))

        # Nothing gathered? bail out
        if all(len(baseline_data[p]) == 0 for p in baseline_data):
            raise RuntimeError("Nothing to plot: selection returned no data.")

        # Colormap and scaling
        cm = cmap or "turbo"

        # For each baseline, plot according to pol selection
        saved = []

        # Union of all baseline IDs observed across pol buckets
        all_baselines = sorted(set().union(*[set(d.keys()) for d in baseline_data.values()]) if baseline_data else [])

        # Helper to render a single panel given entries (list of (t, amp_vec))
        def _render_panel(ax, entries, xlabel, x_min, x_max, log_amp, vmin, vmax, cm, title_suffix):
            from datetime import datetime, timezone
            entries.sort(key=lambda x: x[0])
            times_sorted = [datetime(1858, 11, 17, tzinfo=timezone.utc) + _dt.timedelta(seconds=t) for t, _ in entries]
            amps = np.array([a for _, a in entries], dtype=float)  # (n_rows, nchan_sel)
            plot_mat = np.log10(np.clip(amps, 1e-12, None)) if log_amp else amps

            # Autoscale if needed
            vmin_eff, vmax_eff = vmin, vmax
            if vmin is None or vmax is None:
                finite = np.isfinite(plot_mat)
                if finite.any():
                    vals = plot_mat[finite]
                    lo = np.nanpercentile(vals, 1.0)
                    hi = np.nanpercentile(vals, 99.0)
                    if vmin_eff is None: vmin_eff = lo
                    if vmax_eff is None: vmax_eff = hi

            extent = [x_min, x_max, 0, plot_mat.shape[0]]
            im = ax.imshow(
                plot_mat, aspect="auto", origin="lower",
                extent=extent, vmin=vmin_eff, vmax=vmax_eff, cmap=cm
            )
            ax.set_xlabel(xlabel)
            nt = len(times_sorted)
            yticks = np.linspace(0, nt - 1, min(6, nt), dtype=int) if nt > 0 else []
            ylabels = [times_sorted[i].strftime("%H:%M:%S") for i in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
            ax.set_ylabel("Time [UTC]")
            if title_suffix:
                ax.set_title(title_suffix)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Amplitude [Jy]" + (" (log10)" if log_amp else ""))
            return im

        if isinstance(pol, str) and pol.lower() == "both":
            # Plot *one figure per baseline* with two panels
            for bl_id in all_baselines:
                # collect entries for the first two available pols on this baseline
                pols_available = [p for p in sorted(baseline_data.keys()) if bl_id in baseline_data[p]]
                if len(pols_available) < 2:
                    log.warning(f"Baseline {bl_id}: less than two polarisations found; skipping 'both' layout.")
                    continue
                p0, p1 = pols_available[:2]

                # Build figure with layout
                if layout == "lr":
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
                else:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

                # Left/top: first pol
                title0 = (title or "") + (f"  (pol {p0})")
                _render_panel(axes[0], list(baseline_data[p0][bl_id]), xlabel, x_min, x_max, log_amp, vmin, vmax, cm, title0)

                # Right/bottom: second pol
                title1 = (title or "") + (f"  (pol {p1})")
                _render_panel(axes[1], list(baseline_data[p1][bl_id]), xlabel, x_min, x_max, log_amp, vmin, vmax, cm, title1)

                fig.suptitle(f"Baseline {bl_id}", y=0.995, fontsize=12)
                fig.tight_layout(rect=(0,0,1,0.97))

                # Filename
                if outfile:
                    stem, ext = os.path.splitext(outfile)
                    if not ext:
                        ext = ".png"
                    fname = f"{stem}_polboth_baseline{bl_id.replace('-', '_')}{ext}"
                else:
                    if x_axis == "channel":
                        rng = f"{int(x_min)}-{int(x_max)}ch"
                    else:
                        rng = f"{x_min:.1f}-{x_max:.1f}MHz"
                    scan_tag = (
                        "all" if sel.scan is None else "-".join(map(str, (sel.scan if isinstance(sel.scan, (list, tuple)) else [sel.scan])))
                    )
                    fname = f"waterfall_scans{scan_tag}_polboth_bl{bl_id.replace('-', '_')}_{rng}.png"
                save_path = os.path.join(outdir, fname)
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"[SpecFall] Saved: {save_path}")
                saved.append(save_path)
        else:
            # Single-pol mode: one figure per (pol, baseline) as before
            for p_idx in sorted(baseline_data.keys()):
                bl_dict = baseline_data[p_idx]
                for bl_id in sorted(bl_dict.keys()):
                    entries = bl_dict[bl_id]
                    if not entries:
                        continue
                    fig, ax = plt.subplots(figsize=(10, 5))
                    title_suffix = (title or f"Baseline {bl_id} – Polarisation {p_idx}")
                    _render_panel(ax, list(entries), xlabel, x_min, x_max, log_amp, vmin, vmax, cm, title_suffix)

                    if outfile:
                        stem, ext = os.path.splitext(outfile)
                        if not ext:
                            ext = ".png"
                        fname = f"{stem}_pol{p_idx}_baseline{bl_id.replace('-', '_')}{ext}"
                    else:
                        if x_axis == "channel":
                            rng = f"{int(x_min)}-{int(x_max)}ch"
                        else:
                            rng = f"{x_min:.1f}-{x_max:.1f}MHz"
                        scan_tag = (
                            "all" if sel.scan is None else "-".join(map(str, (sel.scan if isinstance(sel.scan, (list, tuple)) else [sel.scan])))
                        )
                        fname = f"waterfall_scans{scan_tag}_pol{p_idx}_bl{bl_id.replace('-', '_')}_{rng}.png"
                    save_path = os.path.join(outdir, fname)
                    plt.savefig(save_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    print(f"[SpecFall] Saved: {save_path}")
                    saved.append(save_path)

        return


def _resolve_channel_window(meta, sel):
    # If channel window is given, use it; else translate MHz window to channels
    if sel.cmin is not None or sel.cmax is not None:
        c0 = 0 if sel.cmin is None else int(sel.cmin)
        c1 = meta.nchan if sel.cmax is None else int(sel.cmax)
        return max(0, c0), min(meta.nchan, c1)
    if sel.fmin is not None or sel.fmax is not None:
        freqs_mhz = meta.chan_freq / 1e6
        f0 = freqs_mhz[0] if sel.fmin is None else sel.fmin
        f1 = freqs_mhz[-1] if sel.fmax is None else sel.fmax
        c0 = int(np.searchsorted(freqs_mhz, f0, side="left"))
        c1 = int(np.searchsorted(freqs_mhz, f1, side="right"))
        return max(0, c0), min(meta.nchan, c1)
    return 0, meta.nchan


def _resolve_pol(pol):
    if pol is None:
        return slice(None)  # average across all pols
    if isinstance(pol, int):
        return pol
    key = str(pol).lower()
    if key == "both":
        return "both"
    return POL_ALIASES.get(key, 0)


def _amps_for_pol(cube, psel):
    """Return list of 2D arrays (ntime, nchan) per panel from (ntime, nchan, npol).
    - psel == 'both' -> two panels (first two pol indices)
    - psel is slice(None) -> average over pol dimension with NaN-safe mean (no warnings)
    - psel is int -> single selected pol
    """
    if psel == "both":
        outs = []
        for p in range(min(2, cube.shape[-1])):
            outs.append(cube[..., p])
        return outs

    if isinstance(psel, slice):
        # NaN-safe mean over pol axis without runtime warnings for all-NaN rows
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nanmean(cube, axis=-1)
        return [m]

    # integer pol index
    return [cube[..., psel]]


def _apply_mhz_ticks(ax, fmin, fmax, tick=1.0, label_step=5.0):
    """Place minor ticks every `tick` MHz and label only at multiples of `label_step`."""
    f0 = np.ceil(fmin / tick) * tick
    ticks = np.arange(f0, fmax + 1e-9, tick)
    ax.set_xticks(ticks, minor=True)
    majors = [t for t in ticks if np.isclose((t - ticks[0]) % label_step, 0, atol=1e-6)]
    ax.set_xticks(majors)
    ax.set_xticklabels([f"{m:.0f}" for m in majors])


def _save_or_show(fig, plt, outdir, outfile, x_axis, chan_idx, x, sel):
    """Save the figure to disk (with smart defaults) or show it."""
    save = (outdir is not None) or (outfile is not None)
    if save:
        outdir = outdir or "."
        os.makedirs(outdir, exist_ok=True)
        if not outfile:
            # scan tag
            if sel.scan is None:
                scan_tag = "scans_all"
            else:
                s = sel.scan if isinstance(sel.scan, tuple) else tuple(sel.scan)
                scan_tag = "scans_" + "-".join(map(str, s))
            # range tag
            if x_axis == "freq":
                rng_tag = f"{float(x[0]):.1f}-{float(x[-1]):.1f}MHz"
            else:
                rng_tag = f"{int(chan_idx[0])}-{int(chan_idx[-1])}ch"
            outfile = f"waterfall_{scan_tag}_{rng_tag}.png"
        path = os.path.join(outdir, outfile)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[SpecFall] Saved: {path}")
    else:
        plt.show()