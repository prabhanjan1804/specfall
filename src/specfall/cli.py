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

def _parse_baseline_arg(s: str):
    """
    Parse --baseline argument into:
      - 'avg' (str), or
      - [(a,b), (c,d), ...] list of integer tuples.
    Accepts spaces; pairs separated by commas; antennas separated by '-'.
    Examples: 'avg', '3-7', '1-2, 3-7 ,10-11'
    """
    if s is None:
        return "avg"
    s = s.strip().lower()
    if s == "avg":
        return "avg"

    # normalize spaces
    s = s.replace(" ", "")
    if not s:
        return "avg"

    pairs = []
    for part in s.split(","):
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f'Invalid baseline "{part}". Use "a-b" (e.g., 3-7).')
        a, b = part.split("-", 1)
        try:
            a_i = int(a); b_i = int(b)
        except ValueError:
            raise ValueError(f'Baseline IDs must be integers: got "{part}".')
        if a_i == b_i:
            raise ValueError(f"Degenerate baseline {a_i}-{b_i}: antennas must differ.")
        a_i, b_i = sorted((a_i, b_i))
        pairs.append((a_i, b_i))

    return pairs if len(pairs) > 1 else pairs[0]

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
    pw.add_argument("--baseline",default="avg",help='Baseline selection: "avg" | "a-b" | "a-b,c-d,..." (antenna IDs, 0-based).')
    pw.add_argument("--bl-cols",type=int,default=2,help="Number of columns when plotting multiple baselines.")
    pw.add_argument("--bad-bl-only", action="store_true", help="Only plot baselines flagged as bad")
    pw.add_argument("--rms-cut", type=float, default=None,help="RMS threshold for bad baseline detection (Jy)")
    
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
        baseline_sel = _parse_baseline_arg(args.baseline)
        ms.plot.waterfall(
            baseline=baseline_sel,
            bl_cols=args.bl_cols,
            x_axis=args.x_axis,
            log_amp=not args.linear,
            pol=args.pol,
            layout=args.layout,
            cmap=args.cmap,
            outdir=args.outdir,
            outfile=args.outfile,
            bad_bl_only=args.bad_bl_only,
            rms_cut=args.rms_cut,
        )


if __name__ == "__main__":
    main()