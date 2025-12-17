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

import importlib
import inspect


def test_importable():
    specfall = importlib.import_module("specfall")
    assert specfall is not None

def test_plot_namespace():
    specfall = importlib.import_module("specfall")
    assert hasattr(specfall, "plot")
    assert hasattr(specfall.plot, "waterfall")

def test_open_exists():
    specfall = importlib.import_module("specfall")
    assert hasattr(specfall, "open")


def test_waterfall_signature():
    specfall = importlib.import_module("specfall")
    wf = specfall.plot.waterfall

    sig = inspect.signature(wf)

    # Core plotting args
    assert "baseline" in sig.parameters
    assert "outdir" in sig.parameters
    assert "outfile" in sig.parameters
    assert "log_amp" in sig.parameters
    assert "pol" in sig.parameters

    # New functionality
    assert "amp_scale" in sig.parameters
    assert "amp_unit" in sig.parameters

    # Filtering / diagnostics
    assert "bl_cols" in sig.parameters
