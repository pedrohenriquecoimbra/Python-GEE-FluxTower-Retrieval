# fluxsat

Retrieve and extract Google Earth Engine (GEE) satellite data over eddy-covariance flux tower sites, with batch export to local or remote storage.

This is part of ongoing work on high-resolution mapping of greenhouse gas fluxes.

**Author:** Pedro H. Herig Coimbra, researcher at INRAE, France
**Contact:** pedro-henrique.herig-coimbra@inrae.fr

---

## What it does

`fluxsat` pulls satellite-derived variables from Google Earth Engine for the footprint around a flux tower site, so they can be matched against in-situ eddy-covariance measurements. It ships with a configuration for the **FR-Gri** ICOS site as a worked example, and can be pointed at any site by editing the site coordinates and date range.

It can be run two ways:

- **Interactively**, through a Jupyter notebook (`launcher.ipynb`), for exploration and one-off pulls.
- **As a routine**, through a batch script (`launcher.bat`) that runs the extraction and ships the outputs to a server via [Rclone](https://rclone.org/) — or saves them locally.

## Repository layout

| File | Purpose |
|------|---------|
| `ee_google.py` | Core module: authenticates with Earth Engine and retrieves the satellite data for a site. |
| `launcher.ipynb` | Interactive entry point — walks through configuring and running a retrieval. |
| `launcher.bat` | Batch routine for scheduled/automated runs, with optional Rclone upload. |
| `environment.yml` | Conda environment specification. |
| `requirements.txt` | Package list (alternative conda install path). |

## Requirements

- A Google account with [Google Earth Engine access](https://earthengine.google.com/) enabled.
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- (Optional) [Rclone](https://rclone.org/) configured with a remote, if you want automated upload to a server.

## Setup

Clone the repository:

```bash
git clone https://github.com/pedrohenriquecoimbra/fluxsat.git
cd fluxsat
```

Create the environment (choose one):

```bash
# Option 1 — from requirements.txt
conda create -n satsite --file requirements.txt

# Option 2 — from environment.yml
conda env create -f environment.yml
```

Activate it:

```bash
conda activate satsite
```

Authenticate with Earth Engine (one time per machine):

```bash
earthengine authenticate
```

## Usage

### Option A — Interactive (recommended for first run)

Open the notebook and follow it cell by cell:

```bash
jupyter notebook launcher.ipynb
```

The notebook is preconfigured for the FR-Gri ICOS site. To use a different site, edit the site coordinates and the date range near the top of the notebook. <!-- TODO: name the exact variables to change -->

### Option B — Batch routine

Run the routine to extract and push files to your server (or save locally):

```bash
launcher.bat
```

Configure the destination — local path or Rclone remote — inside `launcher.bat` before running. <!-- TODO: document the specific variables / remote name expected -->

## Configuration

<!-- TODO: fill in the specifics that aren't inferable from the file list -->

- **Site location:** [describe the format — e.g. lat/lon, or a site ID]
- **Date range:** [describe how start/end dates are set]
- **Datasets / bands retrieved:** [list the GEE collections and bands pulled, e.g. MODIS LST, Sentinel-2 reflectance, etc.]
- **Output:** [describe the output format and schema — e.g. one CSV per site with columns X, Y, Z]

## Citing

If you use this code in published work, please cite it and/or contact the author. <!-- TODO: add a DOI / citation block if you mint one via Zenodo -->

## License

<!-- TODO: add a license. Without one, the code is "all rights reserved" by default and others can't reuse it. MIT or Apache-2.0 are common for research tooling. -->
