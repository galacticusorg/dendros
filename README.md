# Dendros

<p align="center">
  <img src="https://github.com/galacticusorg/dendros/blob/main/assets/dendros.png?raw=true" width="400" alt="Dendros Logo">
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/dendros.svg)](https://badge.fury.io/py/dendros)
[![Documentation](https://readthedocs.org/projects/dendros/badge/?version=latest)](https://dendros.readthedocs.io/en/latest/)

A Python toolkit for analyzing [Galacticus](https://github.com/galacticusorg/galacticus)
semi-analytic model outputs.

---

## Installation

```bash
pip install dendros
```

To also enable pandas and tabulate table output:

```bash
pip install 'dendros[pandas,tabulate]'
```

To enable plotting of Galacticus `/analyses` results (requires matplotlib):

```bash
pip install 'dendros[plot]'
```

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/galacticusorg/dendros.git
```

---

## Quickstart

### Opening files

```python
from dendros import open_outputs

# Single file
c = open_outputs("galacticus.hdf5")

# Auto-detect MPI-split outputs (given any one rank's file)
c = open_outputs("galacticus_MPI:0000.hdf5")

# Explicit list of files
c = open_outputs(["rank0.hdf5", "rank1.hdf5"])

# Glob pattern
c = open_outputs("run001/galacticus*.hdf5")

# Lightcone run (different top-level group)
c = open_outputs("lightcone.hdf5", output_root="Lightcone")
```

Use `Collection` as a context manager to ensure files are closed:

```python
with open_outputs("galacticus.hdf5") as c:
    ...
```

### Checking completion status

Galacticus writes a `statusCompletion` attribute when a run finishes.
`validate_completion` raises an error if any file is incomplete:

```python
with open_outputs("galacticus.hdf5") as c:
    c.validate_completion()           # raises RuntimeError if incomplete
    c.validate_completion(mode="warn")    # emit warning instead
    c.validate_completion(mode="ignore")  # do nothing
```

### Listing available outputs

```python
with open_outputs("galacticus.hdf5") as c:
    tbl = c.list_outputs()          # astropy Table by default
    print(tbl)

    # or as a pandas DataFrame:
    df = c.list_outputs(format="pandas")

    # or as a tabulate string:
    df = c.list_outputs(format="tabulate")
```

Example output:

```
index  name     time   scale_factor  redshift
----- ------- -------- ------------ ---------
    1 Output1  13.8        1.0          0.0
    2 Output2   6.0        0.5          1.0
```

You can also access the index object directly:

```python
with open_outputs("galacticus.hdf5") as c:
    for meta in c.outputs:
        print(meta.name, meta.redshift)
```

### Listing available properties

```python
with open_outputs("galacticus.hdf5") as c:
    tbl = c.list_properties("Output1")   # by name
    tbl = c.list_properties(1)           # by 1-based integer index
    print(tbl)
```

Example output:

```
name         dtype    shape   description          unitsInSI
---------- ------- -------- -------------------- -----------
haloMass   float64  (1000,) Halo virial mass     1.989e+30
stellarMass float64 (1000,) Stellar mass of disk 1.989e+30
...
```

### Reading datasets

```python
with open_outputs("galacticus.hdf5") as c:
    # List of dataset paths → same strings used as dict keys
    data = c.read("Output1", ["nodeData/basicMass", "nodeData/diskMassStellar"])
    print(data["nodeData/basicMass"])   # numpy array

    # Dict → custom labels
    data = c.read(
        "Output1",
        {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
    )
    print(data["Mhalo"])
```

### Filtering galaxies

Pass a boolean mask or integer index array as `where`:

```python
with open_outputs("galacticus.hdf5") as c:
    # First read to build a mask
    masses = c.read("Output1", ["nodeData/basicMass"])["nodeData/basicMass"]
    mask = masses > 1e12

    # Then read everything for the selected galaxies only
    data = c.read(
        "Output1",
        {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
        where=mask,
    )
```

### h5py-like browsing

```python
with open_outputs("galacticus.hdf5") as c:
    print(c.keys())                        # top-level groups
    grp = c["Outputs/Output1"]
    print(grp.keys())                      # subgroups / datasets
    print(grp.attrs)                       # group attributes
    ds = c["Outputs/Output1/nodeData/basicMass"]
    print(ds.dtype, ds.shape)
```

### Plotting analyses

If a Galacticus run was configured to write reduced analysis results, the
HDF5 file will contain a top-level `/analyses` group with one subgroup per
analysis.  Dendros can list those analyses and plot each model curve with
its observational/target overlay.  Requires the `[plot]` extra.

For MPI runs, the `/analyses` data is reduced over all ranks and is
identical in every rank's file, so dendros reads only the primary file.

```python
with open_outputs("galacticus.hdf5") as c:
    print(c.list_analyses())                     # tabulate available analyses

    figs = c.plot_analyses()                     # one matplotlib Figure per analysis
    figs = c.plot_analyses(name="stellarMassFunction",
                           output_directory="figs",
                           file_format="pdf")    # also save to disk
```

---

## MPI outputs

When Galacticus runs with MPI, it writes one file per rank with the suffix
`_MPI:NNNN` (e.g. `galacticus_MPI:0000.hdf5`, `galacticus_MPI:0001.hdf5`, …).
All ranks contain identical metadata groups; galaxy datasets are split across
ranks.

`open_outputs` handles this automatically:

```python
# Any single-rank file → auto-detects all peers
c = open_outputs("galacticus_MPI:0000.hdf5")

# Or pass an explicit list / glob
c = open_outputs("galacticus_MPI:????.hdf5")
```

`c.read(...)` transparently concatenates arrays across all ranks along axis 0.

---

## Lightcone outputs

For lightcone runs the top-level group is typically `Lightcone` rather than
`Outputs`.  Pass `output_root` to override the default:

```python
c = open_outputs("lightcone.hdf5", output_root="Lightcone")
```

---

## Documentation

Full API reference and more examples are available at
[dendros.readthedocs.io](https://dendros.readthedocs.io).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style,
and how to propose changes.

---

## License

Dendros is released under the
[GNU General Public License v3.0 or later](LICENSE).

