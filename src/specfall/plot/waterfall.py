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
        outdir: str | None = None,     # save directory (if set, we save instead of show)
        outfile: str | None = None,    # optional filename (auto-generated if None)
    ):
        """
        Plot a time×frequency waterfall averaged across baselines.

        If `outdir` or `outfile` is provided, the figure is saved to disk; otherwise shown.
        Averaging: for each timestamp, average |DATA| across baselines at each channel & pol.
        """
        from casacore.tables import table

        meta = self.ms.meta
        sel = self.ms._sel
        pol = pol if pol is not None else sel.pol

        # ----- Determine channel window from freq/channel selection -----
        c0, c1 = _resolve_channel_window(meta, sel)
        chan_idx = np.arange(c0, c1)

        # ----- Determine scans to plot -----
        scans = sel.scan
        if scans is None:
            scans = tuple(np.unique(meta.scans).tolist())
        else:
            scans = tuple(scans)

        # ----- Determine polarization handling -----
        psel = _resolve_pol(pol)

        # ----- Load and aggregate per scan -----
        amps_per_pol = None  # list per pol panel of arrays to concat by time
        times_all = []

        with table(self.ms.path, readonly=True) as T:
            scan_col = T.getcol("SCAN_NUMBER")
            time_col = T.getcol("TIME")
            for sc in scans:
                rows = (scan_col == sc)
                if not np.any(rows):
                    log.warning(f"No rows for scan {sc}; skipping.")
                    continue
                r_idx = np.where(rows)[0]
                start, count = int(r_idx[0]), int(rows.sum())

                data = T.getcol(meta.data_col, startrow=start, nrow=count)  # (count, nchan, npol)
                flag = T.getcol("FLAG", startrow=start, nrow=count)         # (count, nchan, npol)
                times = time_col[start:start+count]                          # (count,)

                # cut to channel window
                data = data[:, c0:c1, :]
                flag = flag[:, c0:c1, :]

                # magnitude with flags masked
                amp = np.abs(data)
                amp = np.where(~flag, amp, np.nan)

                # group by unique times and average across baselines for each time
                t_unique, inv = np.unique(times, return_inverse=True)
                ntime = t_unique.size
                nchan = c1 - c0
                npol = amp.shape[-1]
                out = np.empty((ntime, nchan, npol), dtype=float)
                out[:] = np.nan

                sums = np.zeros_like(out)
                counts = np.zeros_like(out)
                idx = inv[:, None, None]
                valid = ~np.isnan(amp)
                vals = np.where(valid, amp, 0.0)
                np.add.at(sums, (idx, np.s_[:], np.s_[:]), vals)
                np.add.at(counts, (idx, np.s_[:], np.s_[:]), valid.astype(float))
                out = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

                # prepare panels
                panels = _amps_for_pol(out, psel)
                if amps_per_pol is None:
                    amps_per_pol = [[] for _ in range(len(panels))]
                for p_i, panel in enumerate(panels):
                    amps_per_pol[p_i].append(panel)  # (ntime, nchan)
                times_all.append(t_unique)

        if not times_all:
            raise RuntimeError("Nothing to plot: selection returned no data.")

        # ----- Concatenate along time axis -----
        times = np.concatenate(times_all)
        t0 = times.min()
        thours = (times - t0) / 3600.0
        # Build per-panel matrices
        mats = [np.concatenate(chunks, axis=0) for chunks in amps_per_pol]  # (NT, NC)

        # ----- X axis -----
        if x_axis == "channel":
            x = chan_idx
            xlabel = "Channel"
        else:
            x = (self.ms.meta.chan_freq[chan_idx] / 1e6)  # MHz
            xlabel = "Frequency [MHz]"

        # ----- Set up figure -----
        n_panels = len(mats)
        if n_panels == 2 and layout == "lr":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        elif n_panels == 2:
            fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(9, 5))
            axes = np.atleast_1d(axes)

        # ----- Plot each panel -----
        for p, ax in enumerate(axes):
            mat = mats[p]
            plot_mat = np.log10(np.clip(mat, 1e-12, None)) if log_amp else mat
            im = ax.imshow(
                plot_mat,
                aspect="auto",
                origin="lower",
                extent=[x[0], x[-1], thours.min(), thours.max()],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_ylabel("Time [hr from start]")
            if p == len(axes) - 1:
                ax.set_xlabel(xlabel)
            ax.set_title(title or f"Waterfall ({'log' if log_amp else 'lin'})")

            # Optional MHz ticks helper
            if x_axis == "freq" and mhz_tick:
                _apply_mhz_ticks(ax, x.min(), x.max(), tick=mhz_tick, label_step=mhz_label or mhz_tick)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Amplitude" + (" (log10)" if log_amp else ""))

        fig.tight_layout()

        # ----- SAVE or SHOW -----
        save = (outdir is not None) or (outfile is not None)
        if save:
            outdir = outdir or "."
            os.makedirs(outdir, exist_ok=True)

            # auto filename if none provided
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