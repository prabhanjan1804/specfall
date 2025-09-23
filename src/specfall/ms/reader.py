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
from dataclasses import dataclass
from .selection import Selection
from ..utils.checks import require
from ..utils.logging import log

@dataclass(frozen=True)
class Meta:
    nrow: int
    nchan: int
    npol: int
    scans: np.ndarray  # (nrow,) per-row scan number
    times: np.ndarray  # (nrow,) seconds (MJD seconds)
    chan_freq: np.ndarray  # (nchan,) Hz
    data_col: str  # "CORRECTED_DATA" if exists else "DATA"

class MeasurementSet:
    """A lightweight, chainable MS handle with a selection view and plotting namespace."""

    def __init__(self, path: str):
        require("casacore.tables", hint="Install python-casacore (conda-forge).")
        from casacore.tables import table

        self.path = path
        with table(path, readonly=True) as T:
            cols = set(T.colnames())
            data_col = "CORRECTED_DATA" if "CORRECTED_DATA" in cols else "DATA"
            # Grab shapes: getcol returns shape (nrow, nchan, npol)
            nrow = T.nrows()
            # Avoid loading big arrays just to get shape: read one row
            sample = T.getcell(data_col, 0)  # (nchan, npol)
            nchan, npol = sample.shape
            scans = T.getcol("SCAN_NUMBER")  # (nrow,)
            times = T.getcol("TIME")         # (nrow,)

        with table(f"{path}::SPECTRAL_WINDOW", readonly=True) as SPW:
            chan_freq = SPW.getcol("CHAN_FREQ")[0]  # (nchan,)

        self._meta = Meta(
            nrow=nrow,
            nchan=nchan,
            npol=npol,
            scans=scans,
            times=times,
            chan_freq=chan_freq,
            data_col=data_col,
        )
        self._sel = Selection()  # default: all
        log.info(
            f"Opened MS: rows={nrow}, nchan={nchan}, npol={npol}, data_col={data_col}"
        )

    # ---------- Selection API ----------
    @property
    def meta(self) -> Meta:
        return self._meta

    def select(
        self,
        scan: int | list[int] | tuple[int, ...] | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        cmin: int | None = None,
        cmax: int | None = None,
        pol: str | int | None = None,
    ) -> "MeasurementSet":
        new = MeasurementSet.__new__(MeasurementSet)
        new.path = self.path
        new._meta = self._meta
        # normalize scan to tuple
        if isinstance(scan, int):
            scan = (scan,)
        elif isinstance(scan, list):
            scan = tuple(scan)
        new._sel = self._sel.updated(scan=scan, fmin=fmin, fmax=fmax, cmin=cmin, cmax=cmax, pol=pol)
        return new

    # Plotting namespace
    @property
    def plot(self):
        from ..plot.waterfall import WaterfallPlotter
        return WaterfallPlotter(self)