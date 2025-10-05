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

# src/specfall/utils/checks.py
import importlib
try:
    import importlib.util as _importlib_util
except Exception:
    _importlib_util = None

def require(module: str, hint: str = ""):
    """Ensure a module is importable; raise a clear error otherwise."""
    has = False
    if _importlib_util is not None:
        try:
            has = _importlib_util.find_spec(module) is not None
        except Exception:
            has = False
    if not has:
        try:
            __import__(module)
            has = True
        except Exception:
            has = False
    if not has:
        msg = f"Missing optional dependency: {module}"
        if hint:
            msg += f"\nHint: {hint}"
        raise ImportError(msg)