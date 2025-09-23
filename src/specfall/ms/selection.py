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
from dataclasses import dataclass, replace
from typing import Tuple

@dataclass(frozen=True)
class Selection:
    """Immutable selection applied to an MS view."""
    scan: Tuple[int, ...] | None = None
    fmin: float | None = None  # MHz
    fmax: float | None = None  # MHz
    cmin: int | None = None
    cmax: int | None = None
    pol: str | int | None = None  # "both", 0, 1, "XX", "YY", etc.

    def updated(self, **kwargs) -> "Selection":
        """Return a new Selection with provided (non-None) fields replaced."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})