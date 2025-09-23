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
based on the scan and frequency/channel range.
Following are examples, please adapt paths and parameters as needed.

### Python

```python
import specfall as sf

ms = sf.open("/data/target.ms")

# Quick default plot (all scans, full band, log amplitude)
# → shown interactively
ms.plot.waterfall()

# Save default plot to "plots/waterfall.png"
ms.plot.waterfall(outdir="plots")

# Selected scan, frequency window, linear amplitude, plasma colormap
# → save with a custom filename
ms.select(scan=3, fmin=1355.0, fmax=1382.0).plot.waterfall(
    log_amp=False,
    cmap="plasma",
    outdir="results",
    outfile="scan3.png"
)

# Use channels on X-axis, both polarisations top–bottom, inferno colormap
# → auto-generate filename inside "outputs/"
ms.select(scan=[1, 2]).plot.waterfall(
    x_axis="channel",
    pol="both",
    layout="tb",
    cmap="inferno",
    outdir="outputs"
)
```

### Command Line Interface

```bash
# Default (interactive show)
specfall plot /path/to/data.ms

# Save default plot into "plots/waterfall.png"
specfall plot /path/to/data.ms --outdir plots

# Specific scan and frequency window, plasma colormap
# → save to results/scan3.png
specfall plot /path/to/data.ms --scan 3 --freq 1355:1382 --cmap plasma --outdir results --outfile scan3.png

# Two scans, channel range, both pols stacked, inferno colormap
# → auto filename generated inside "outputs/"
specfall plot /path/to/data.ms --scan 1 2 --chan 100:600 --pol both --layout tb --cmap inferno --outdir outputs
```


## Notes
- If `CORRECTED_DATA` is absent, SpecFall automatically falls back to using `DATA`.  
- Frequency tick helper: default ticks are placed every **1 MHz**, with labels every **5 MHz**.  
- For very large datasets, consider **chunked reading** or integrating with **Dask** for scalable performance.  
- On **Windows**, `python-casacore` is not natively supported.  
  - Recommended: run SpecFall inside **WSL2 (Ubuntu)** for full functionality.  
  - Advanced users may attempt a manual CASACORE + python-casacore build, but this is not officially supported.  
- Plots are designed as **diagnostic quick–looks**: SpecFall averages over baselines per timestamp for speed and clarity.  
- Any valid **Matplotlib colormap** can be used (e.g. `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `gray`, `jet`). 

## License
SpecFall is distributed under GNU GENERAL PUBLIC LICENSE v3
Copyright (C) (2025) Prabhanjan H. Kulkarni
