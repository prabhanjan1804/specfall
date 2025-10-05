# SpecFall

**SpecFall** is a lightweight Python package for quick–look visualization of **radio interferometric data** stored in CASA **Measurement Sets (.ms)**.  

It provides diagnostic *waterfall plots* with:  
- **Frequency or channel** on the horizontal axis  
- **Time** on the vertical axis  
- **Amplitude** mapped to a color scale  

SpecFall is designed for rapid inspection of raw or calibrated visibilities, helping identify instrumental effects, RFI, and general data quality issues.

---

## Features
- Flexible axis selection: **frequency** (MHz) or **channel number**  
- Amplitude scaling: **linear** or **logarithmic**  
- Scan selection: plot full dataset or chosen scans only  
- Frequency/channel windowing for zoomed analysis  
- Polarisation handling: single pol or both pols (arranged top–bottom or side–by–side)  
- Choice of any Matplotlib colormap (e.g. *viridis, plasma, inferno, cividis, gray, jet*)  
- Simple **CLI** for batch jobs (suitable for SLURM/HPC environments)  
- Pure Python API for integration into custom workflows  

---

## Installation

### For MacOs & Linux

Clone from GitHub and install in editable mode:

```bash
git clone https://github.com/prabhanjan1804/specfall.git
cd specfall
pip install -e .
```
Or Install directly via pip

```bash
pip install --upgrade git+https://github.com/prabhanjan1804/specfall.git
```
**Dependencies**: numpy, matplotlib, and python-casacore

### For Windows
SpecFall depends on python-casacore, which in turn requires the CASA Core C++ libraries. These libraries are not officially supported on Windows, so you cannot install python-casacore directly with pip on a native Windows environment.

#### WSL 2 (Windows Subsystem for Linux) {Recommended}
Install WSL2 for Windows
```bash
sudo apt update && sudo apt install -y git python3 python3-pip
pip install numpy matplotlib
pip install --upgrade git+https://github.com/prabhanjan1804/SpecFall.git
```

## Usage

Both the Python API and the CLI support interactive display (`plt.show()`)  
or saving plots to disk using `outdir` and `outfile`.  
If only `outdir` is specified, SpecFall will automatically generate a filename  
based on the scan, baseline, and frequency/channel range.

New in v0.1.1:  
SpecFall now supports **baseline-wise plotting** — users can view amplitude as a function of time and frequency **per baseline** rather than averaging across all baselines.

### Python

```python
import specfall as sf

ms = sf.open("/data/target.ms")

# Quick default plot (all scans, full band, averaged over baselines)
ms.plot.waterfall(outdir="plots")

# Plot a specific baseline (e.g., antennas 2–5)
ms.select(scan=2).plot.waterfall(
    baseline=(2, 5),
    cmap="inferno",
    outdir="results"
)

# Plot multiple baselines (e.g., [(0,1), (0,2), (1,2)])
ms.select(scan=1).plot.waterfall(
    baseline=[(0, 1), (0, 2), (1, 2)],
    cmap="plasma",
    outdir="results_multi"
)

# Save per-baseline plots with a custom filename prefix
ms.select(scan=2).plot.waterfall(
    baseline="avg",          # average across all baselines
    log_amp=False,
    cmap="viridis",
    outdir="plots",
    outfile="scan2"
)
```

### Command Line Interface

```bash
# Default (averaged over baselines)
specfall plot /path/to/data.ms --outdir plots

# Specific baseline (antennas 2–5)
specfall plot /path/to/data.ms --scan 2 --baseline 2 5 --cmap inferno --outdir results

# Multiple baselines
specfall plot /path/to/data.ms --scan 1 --baseline 0 1 0 2 1 2 --cmap plasma --outdir results_multi

# Average over all baselines, custom prefix
specfall plot /path/to/data.ms --scan 2 --baseline avg --log-amp False --cmap viridis --outdir plots --outfile scan2
```

---

## Notes
- If `CORRECTED_DATA` is absent, SpecFall automatically falls back to using `DATA`.  
- Frequency tick helper: default ticks are placed every **1 MHz**, with labels every **5 MHz**.  
- For very large datasets, consider **chunked reading** or integrating with **Dask** for scalable performance.  
- On **Windows**, `python-casacore` is not natively supported.  
  - Recommended: run SpecFall inside **WSL2 (Ubuntu)** for full functionality.  
  - Advanced users may attempt a manual CASACORE + python-casacore build, but this is not officially supported.  
- Plots are designed as **diagnostic quick–looks**: SpecFall averages over baselines per timestamp for speed and clarity.  
- Any valid **Matplotlib colormap** can be used (e.g. `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `gray`, `jet`).  
- Baseline selection: `'avg'` (default), single tuple `(ant1, ant2)`, or list of pairs.  
- Each baseline is saved as a separate PNG with Y-axis showing UTC timestamps.

## License
SpecFall is distributed under GNU GENERAL PUBLIC LICENSE v3
Copyright (C) (2025) Prabhanjan H. Kulkarni
