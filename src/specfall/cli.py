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

import argparse
from .api import open as ms_open


def main(argv=None):
    p = argparse.ArgumentParser(prog="specfall")
    sub = p.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("plot", help="Waterfall plot")
    pw.add_argument("ms", help="Path to .ms")
    pw.add_argument("--scan", type=int, nargs="*", help="Scan number(s)")
    pw.add_argument("--freq", help="MHz window fmin:fmax")
    pw.add_argument("--chan", help="Channel window cmin:cmax")
    pw.add_argument("--x-axis", choices=["freq", "channel"], default="freq")
    pw.add_argument("--linear", action="store_true", help="Use linear amplitude")
    pw.add_argument("--pol", default=None, help="0,1,XX,YY,both")
    pw.add_argument("--layout", choices=["tb", "lr"], default="tb")
    pw.add_argument("--outdir", help="Directory to save plot instead of showing")
    pw.add_argument("--outfile", help="Optional filename for saved plot (default: waterfall.png)")

    args = p.parse_args(argv)

    if args.cmd == "plot":
        ms = ms_open(args.ms)
        sel = {}
        if args.scan:
            sel["scan"] = args.scan if len(args.scan) > 1 else args.scan[0]
        if args.freq:
            fmin, fmax = map(float, args.freq.split(":"))
            sel.update({"fmin": fmin, "fmax": fmax})
        if args.chan:
            cmin, cmax = map(int, args.chan.split(":"))
            sel.update({"cmin": cmin, "cmax": cmax})
        ms = ms.select(**sel)
        ms.plot.waterfall(
            x_axis=args.x_axis,
            log_amp=not args.linear,
            pol=args.pol,
            layout=args.layout,
            cmap=args.cmap,
            outdir=args.outdir,
            outfile=args.outfile,
        )


if __name__ == "__main__":
    main()