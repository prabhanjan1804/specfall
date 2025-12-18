<file name=0 path=/Users/phk/specfall/README.md># SpecFall


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
- Automatic RMS calculation per baseline
- Optional filtering to plot only baselines exceeding a user-defined RMS threshold
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

New in v1.0.0:  
SpecFall supports baseline-wise waterfall plotting with **automatic RMS computation and RMS-based baseline filtering**, allowing users to identify and visualise only those baselines exhibiting anomalously high noise levels in their time–frequency amplitude statistics.

### Getting Help

SpecFall provides a built-in help system that documents all available plotting
options, default values, and diagnostic features.

From Python:
```python
ms.plot.help()
```

From the command line:
```bash
specfall help
```

This prints a detailed overview of axis options, baseline handling, RMS-based
filtering, polarisation layouts, and output settings.

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

# Plot only baselines with RMS above a threshold (Jy)
ms.select(scan=2).plot.waterfall(
    rms_cut=5.0,
    bad_bl_only=True,
    cmap="plasma",
    outdir="bad_baselines"
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

# Plot only baselines exceeding an RMS threshold
specfall plot /path/to/data.ms --scan 2 --rms-cut 5.0 --bad-bl-only --outdir bad_baselines

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
- By default, SpecFall averages visibilities per baseline per timestamp; baseline averaging across the array is optional.
- RMS is computed per baseline from the plotted time–frequency waterfall data and expressed in Jansky.
- Baseline filtering is optional and disabled by default.
- When filtering is enabled, only baselines exceeding the RMS threshold are visualized, reducing the number of output images for large datasets.
- The tool is intended for diagnostic inspection rather than calibration.
- Any valid **Matplotlib colormap** can be used (e.g. `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `gray`, `jet`).  
- Baseline selection: `'avg'` (default), single tuple `(ant1, ant2)`, or list of pairs.  
- Each baseline is saved as a separate PNG with Y-axis showing UTC timestamps.
- Use `ms.plot.help()` or `specfall help` to view a complete, up-to-date description
  of all plotting options and their default settings.

## License
SpecFall is distributed under GNU GENERAL PUBLIC LICENSE v3
Copyright (C) (2025) Prabhanjan H. Kulkarni
</file>
