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
from ..utils.logging import log
import os

POL_ALIASES = {"xx": 0, "yy": 1, "rr": 0, "ll": 1, "xy": 0, "yx": 1}

class WaterfallPlotter:
    def __init__(self, ms):
        self.ms = ms

    def waterfall(
        self,
        x_axis: str = "freq",          # "freq" or "channel"
        log_amp: bool = True,
        pol: str | int | None = None,  # override selection
        layout: str = "tb",            # when pol="both": "tb" or "lr"
        vmax: float | None = None,
        vmin: float | None = None,
        cmap: str = "viridis",
        title: str | None = None,
        mhz_tick: float | None = 1.0,  # tick every 1 MHz (if x=freq)
        mhz_label: float | None = 5.0, # label every 5 MHz (if x=freq)
        outdir: str | None = None,     # save dir (if set, save instead of show)
        outfile: str | None = None,    # optional filename (auto if None)
        baseline: str | tuple[int, int] | list[tuple[int, int]] = "avg",
        bl_cols: int = 2,              # grid columns for multi-baseline
    ):
        """
        Plot time×frequency waterfalls from MS data.

        baseline:
          - "avg": average across all baselines (per timestamp) [default]
          - (a1, a2): single baseline by antenna IDs (order-insensitive)
          - [(a1,a2), (a3,a4), ...]: multiple baselines (grid of panels)

        If outdir or outfile is provided, saves the figure; otherwise shows it.
        """
        from casacore.tables import table

        meta = self.ms.meta
        sel = self.ms._sel
        pol = pol if pol is not None else sel.pol

        # ----- channel window from selection -----
        c0, c1 = _resolve_channel_window(meta, sel)
        chan_idx = np.arange(c0, c1)

        # ----- scans to plot -----
        scans = sel.scan
        if scans is None:
            scans = tuple(np.unique(meta.scans).tolist())
        else:
            scans = tuple(scans)

        # ----- polarization handling -----
        psel = _resolve_pol(pol)

        # ----- normalize baseline selection -----
        # returns: "avg" | set of (min(a1,a2), max(a1,a2)) tuples
        def _normalize_bl(blspec):
            if blspec == "avg" or blspec is None:
                return "avg"
            if isinstance(blspec, tuple) and len(blspec) == 2:
                a, b = sorted(map(int, blspec))
                return {(a, b)}
            if isinstance(blspec, list):
                out = set()
                for t in blspec:
                    if not (isinstance(t, tuple) and len(t) == 2):
                        raise ValueError("Each baseline tuple must be (ant1, ant2)")
                    a, b = sorted(map(int, t))
                    out.add((a, b))
                return out
            raise ValueError('baseline must be "avg", (a1,a2), or [(a1,a2), ...]')

        blspec = _normalize_bl(baseline)

        # ----- read rows and group by baseline -----
        # Build dict: bl -> list of (times, cube[ntime,nchan,npanels]) chunks (per scan)
        by_bl: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}

        with table(self.ms.path, readonly=True) as T:
            scan_col = T.getcol("SCAN_NUMBER")
            time_col = T.getcol("TIME")
            a1_col   = T.getcol("ANTENNA1")
            a2_col   = T.getcol("ANTENNA2")

            for sc in scans:
                rows = (scan_col == sc)
                if not np.any(rows):
                    log.warning(f"No rows for scan {sc}; skipping.")
                    continue
                r_idx = np.where(rows)[0]
                start, count = int(r_idx[0]), int(rows.sum())

                # read columns for this scan chunk
                times = time_col[start:start+count]                          # (count,)
                a1    = a1_col[start:start+count]                             # (count,)
                a2    = a2_col[start:start+count]                             # (count,)
                data  = T.getcol(meta.data_col, startrow=start, nrow=count)   # (count, nchan, npol)
                flag  = T.getcol("FLAG",      startrow=start, nrow=count)     # (count, nchan, npol)

                # channel cut
                data = data[:, c0:c1, :]
                flag = flag[:, c0:c1, :]

                # magnitude with flags masked
                amp = np.abs(data)
                amp = np.where(~flag, amp, np.nan)                            # (count, nchan, npol)

                # split rows by baseline
                bl_keys = np.stack([np.minimum(a1, a2), np.maximum(a1, a2)], axis=1)  # (count,2)
                # unique baselines present in this scan
                uniq_bls, inv = np.unique(bl_keys, axis=0, return_inverse=True)
                for bi, (ba, bb) in enumerate(uniq_bls):
                    bl = (int(ba), int(bb))
                    # if user asked for specific baselines, skip others
                    if blspec != "avg" and bl not in blspec:
                        continue
                    sel_rows = (inv == bi)
                    if not np.any(sel_rows):
                        continue
                    # rows for this baseline
                    amp_bl   = amp[sel_rows]         # (r_b, nchan, npol)
                    time_bl  = times[sel_rows]       # (r_b,)

                    # group by unique times for this baseline
                    t_unique, inv_t = np.unique(time_bl, return_inverse=True)
                    ntime = t_unique.size
                    nchan = c1 - c0
                    npoln = amp_bl.shape[-1]
                    out = np.empty((ntime, nchan, npoln), dtype=float); out[:] = np.nan
                    sums = np.zeros_like(out)
                    cnts = np.zeros_like(out)
                    idx = inv_t[:, None, None]
                    valid = ~np.isnan(amp_bl)
                    vals  = np.where(valid, amp_bl, 0.0)
                    np.add.at(sums, (idx, np.s_[:], np.s_[:]), vals)
                    np.add.at(cnts, (idx, np.s_[:], np.s_[:]), valid.astype(float))
                    out = np.divide(sums, cnts, out=np.full_like(sums, np.nan), where=cnts > 0)

                    # panels by pol (list of 2D arrays)
                    panels = _amps_for_pol(out, psel)   # list of (ntime, nchan)
                    if bl not in by_bl:
                        by_bl[bl] = []
                    by_bl[bl].append((t_unique, np.stack(panels, axis=-1)))  # (ntime, nchan, npanels)

        if blspec == "avg":
            # average/concat across all baselines (quick-look)
            times_all = []
            panels_all = []
            for _, chunks in by_bl.items():
                for (t_u, pan_cube) in chunks:
                    times_all.append(t_u)
                    panels_all.append(pan_cube)  # (ntime, nchan, npanels)
            if not times_all:
                raise RuntimeError("Nothing to plot: selection returned no data.")

            times = np.concatenate(times_all)
            t0 = times.min()
            thours = (times - t0) / 3600.0

            pan_concat = np.concatenate(panels_all, axis=0)  # (NT, NC, NP)
            mats = [pan_concat[..., i] for i in range(pan_concat.shape[-1])]

            # X axis
            if x_axis == "channel":
                x = chan_idx; xlabel = "Channel"
            else:
                x = (self.ms.meta.chan_freq[chan_idx] / 1e6); xlabel = "Frequency [MHz]"

            # Figure layout
            n_panels = len(mats)
            if n_panels == 2 and layout == "lr":
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            elif n_panels == 2:
                fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
            else:
                fig, axes = plt.subplots(1, 1, figsize=(9, 5)); axes = np.atleast_1d(axes)

            for p, ax in enumerate(axes):
                plot_mat = np.log10(np.clip(mats[p], 1e-12, None)) if log_amp else mats[p]
                im = ax.imshow(plot_mat, aspect="auto", origin="lower",
                               extent=[x[0], x[-1], thours.min(), thours.max()],
                               vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_ylabel("Time [hr from start]")
                if p == len(axes) - 1:
                    ax.set_xlabel(xlabel)
                ttl = title or f"Waterfall avg ({'log' if log_amp else 'lin'})"
                ax.set_title(ttl)
                if x_axis == "freq" and mhz_tick:
                    _apply_mhz_ticks(ax, x.min(), x.max(), tick=mhz_tick, label_step=mhz_label or mhz_tick)
                cbar = plt.colorbar(im, ax=ax); cbar.set_label("Amplitude" + (" (log10)" if log_amp else ""))

            fig.tight_layout()

            # save/show
            _save_or_show(fig, plt, outdir, outfile, x_axis, chan_idx, x, sel)
            return

        # ----- Specific baseline(s): build per-baseline panels -----
        # Merge chunks per baseline along time
        bl_panels: dict[tuple[int, int], dict[str, list[np.ndarray] | np.ndarray]] = {}
        for bl, chunks in by_bl.items():
            times = np.concatenate([t for (t, _) in chunks], axis=0)
            mats3 = np.concatenate([m for (_, m) in chunks], axis=0)  # (NT, NC, NP)
            bl_panels[bl] = {"times": times, "mats": [mats3[..., i] for i in range(mats3.shape[-1])]}

        if not bl_panels:
            raise RuntimeError("Nothing to plot for requested baseline(s).")

        # X axis
        if x_axis == "channel":
            x = chan_idx; xlabel = "Channel"
        else:
            x = (self.ms.meta.chan_freq[chan_idx] / 1e6); xlabel = "Frequency [MHz]"

        # Grid for multiple baselines
        bl_list = sorted(bl_panels.keys())
        n_bl = len(bl_list)
        ncols = min(max(1, bl_cols), n_bl)
        nrows = int(np.ceil(n_bl / ncols))

        fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
        gs = fig.add_gridspec(nrows, ncols, wspace=0.25, hspace=0.35)

        for i, bl in enumerate(bl_list):
            r, c = divmod(i, ncols)
            sub_gs = gs[r, c]

            # axes per baseline cell
            if len(bl_panels[bl]["mats"]) == 2:
                if layout == "lr":
                    inner = sub_gs.subgridspec(1, 2, wspace=0.1)
                    axes = np.array([fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])])
                else:
                    inner = sub_gs.subgridspec(2, 1, hspace=0.1)
                    axes = np.array([fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0])])
            else:
                axes = np.array([fig.add_subplot(sub_gs)])

            times = bl_panels[bl]["times"]
            thours = (times - times.min()) / 3600.0
            mats = bl_panels[bl]["mats"]

            for p, ax in enumerate(axes):
                plot_mat = np.log10(np.clip(mats[p], 1e-12, None)) if log_amp else mats[p]
                im = ax.imshow(plot_mat, aspect="auto", origin="lower",
                               extent=[x[0], x[-1], thours.min(), thours.max()],
                               vmin=vmin, vmax=vmax, cmap=cmap)
                if p == 0:
                    ax.set_title(f"BL {bl[0]}–{bl[1]}  " + (title or ""))
                ax.set_ylabel("Time [hr]")
                if (layout == "tb" and p == len(axes) - 1) or (layout == "lr" and p == len(axes) - 1):
                    ax.set_xlabel(xlabel)
                if x_axis == "freq" and mhz_tick:
                    _apply_mhz_ticks(ax, x.min(), x.max(), tick=mhz_tick, label_step=mhz_label or mhz_tick)
                cbar = plt.colorbar(im, ax=ax); cbar.set_label("Amp" + (" (log10)" if log_amp else ""))

        fig.suptitle(title or "Waterfall per baseline", y=0.995, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        _save_or_show(fig, plt, outdir, outfile, x_axis, chan_idx, x, sel)


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
    """Return list of 2D arrays (ntime, nchan) per panel to plot from (ntime, nchan, npol)."""
    if psel == "both":
        outs = []
        for p in range(min(2, cube.shape[-1])):
            outs.append(cube[..., p])
        return outs
    if isinstance(psel, slice):
        return [np.nanmean(cube, axis=-1)]
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