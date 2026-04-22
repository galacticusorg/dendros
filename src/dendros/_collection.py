"""Core collection abstraction and ``open_outputs`` entry point."""
from __future__ import annotations

import glob as _glob
import re
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import h5py
import numpy as np

from ._outputs import OutputIndex

_MPI_SUFFIX = re.compile(r"^(.+):MPI(\d{4})$")


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def _decode(value) -> str:
    """Decode bytes/numpy-string HDF5 attribute values to ``str``."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    # numpy.bytes_ is a subclass of bytes; numpy.str_ is a subclass of str
    return str(value) if not isinstance(value, str) else value


def _resolve_output_name(output: Union[str, int]) -> str:
    """Return the Output* group name for a string or 1-based integer."""
    if isinstance(output, int):
        return f"Output{output}"
    return output


# ---------------------------------------------------------------------------
# Proxy objects – lightweight h5py-like wrappers
# ---------------------------------------------------------------------------


class GroupProxy:
    """Read-only h5py-like proxy for an HDF5 group.

    Parameters
    ----------
    collection:
        Parent :class:`Collection`.
    path:
        HDF5 path to the group within the file.
    """

    def __init__(self, collection: "Collection", path: str) -> None:
        self._collection = collection
        self._path = path

    def _group(self) -> h5py.Group:
        return self._collection._primary[self._path]

    @property
    def attrs(self) -> dict:
        """Return group attributes as a plain dict."""
        return {k: _decode(v) for k, v in self._group().attrs.items()}

    @property
    def name(self) -> str:
        """HDF5 path of this group."""
        return self._path

    def keys(self) -> List[str]:
        """Return the immediate children of this group."""
        return list(self._group().keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self._group().keys())

    def __contains__(self, key: str) -> bool:
        return key in self._group()

    def __getitem__(self, key: str) -> Union["GroupProxy", "DatasetProxy"]:
        path = self._path.rstrip("/") + "/" + key.lstrip("/")
        item = self._collection._primary[path]
        if isinstance(item, h5py.Group):
            return GroupProxy(self._collection, path)
        return DatasetProxy(self._collection, path)

    def __repr__(self) -> str:
        return f"<GroupProxy '{self._path}'>"


class DatasetProxy:
    """Read-only h5py-like proxy for an HDF5 dataset.

    For multi-file :class:`Collection` instances, :meth:`read` concatenates
    data from all files along axis 0.

    Parameters
    ----------
    collection:
        Parent :class:`Collection`.
    path:
        HDF5 path to the dataset within the file.
    """

    def __init__(self, collection: "Collection", path: str) -> None:
        self._collection = collection
        self._path = path

    def _dataset(self) -> h5py.Dataset:
        return self._collection._primary[self._path]

    @property
    def attrs(self) -> dict:
        """Return dataset attributes as a plain dict."""
        return {k: _decode(v) for k, v in self._dataset().attrs.items()}

    @property
    def dtype(self):
        """NumPy dtype of the dataset."""
        return self._dataset().dtype

    @property
    def shape(self) -> tuple:
        """Total shape; for multi-file collections axis-0 is the sum across files."""
        shapes = [h[self._path].shape for h in self._collection._handles]
        if not shapes:
            return ()
        if len(shapes) == 1:
            return shapes[0]
        total = sum(s[0] for s in shapes)
        return (total,) + shapes[0][1:]

    @property
    def name(self) -> str:
        """HDF5 path of this dataset."""
        return self._path

    def read(self, where=None) -> np.ndarray:
        """Read the dataset into a :class:`numpy.ndarray`.

        For multi-file collections the arrays from all files are concatenated
        along axis 0 before the optional *where* selection is applied.

        Parameters
        ----------
        where:
            ``None`` reads everything.  A boolean mask or integer index array
            is applied after concatenation.
        """
        arrays = [h[self._path][()] for h in self._collection._handles]
        arr = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        if where is not None:
            arr = arr[np.asarray(where)]
        return arr

    def __getitem__(self, selection) -> np.ndarray:
        """Index into the dataset of the primary file (h5py-like)."""
        return self._dataset()[selection]

    def __repr__(self) -> str:
        return f"<DatasetProxy '{self._path}' dtype={self.dtype} shape={self.shape}>"


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class Collection:
    """A collection of one or more Galacticus HDF5 output files.

    Prefer constructing instances through :func:`open_outputs` rather than
    calling this constructor directly.

    Parameters
    ----------
    files:
        Paths to the HDF5 files to open (read-only).
    output_root:
        Name of the top-level HDF5 group that contains the ``Output*`` subgroups.
        Defaults to ``"Outputs"``.

    Examples
    --------
    >>> from dendros import open_outputs
    >>> with open_outputs("galacticus.hdf5") as c:
    ...     c.validate_completion()
    ...     print(c.list_outputs())
    ...     data = c.read("Output1", ["nodeData/haloMass"])
    """

    def __init__(self, files: List[str], output_root: str = "Outputs") -> None:
        self._files: List[str] = list(files)
        self._output_root: str = output_root
        self._handles: List[h5py.File] = [h5py.File(f, "r") for f in self._files]
        self._outputs_index: Optional[OutputIndex] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _primary(self) -> h5py.File:
        """The first open file handle; used for metadata and structure."""
        return self._handles[0]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def files(self) -> List[str]:
        """Paths of the files in this collection (in order)."""
        return list(self._files)

    @property
    def output_root(self) -> str:
        """Top-level HDF5 group containing the ``Output*`` groups."""
        return self._output_root

    # ------------------------------------------------------------------
    # h5py-like interface
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """Return the top-level group keys of the primary file."""
        return list(self._primary.keys())

    def __getitem__(self, key: str) -> Union[GroupProxy, DatasetProxy]:
        """Access a group or dataset by HDF5 path (h5py-like)."""
        item = self._primary[key]
        if isinstance(item, h5py.Group):
            return GroupProxy(self, key)
        return DatasetProxy(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._primary

    def __repr__(self) -> str:
        n = len(self._files)
        return (
            f"<Collection files={n} output_root='{self._output_root}'"
            f" path='{self._files[0]}'>"
        )

    # ------------------------------------------------------------------
    # Completion validation
    # ------------------------------------------------------------------

    def validate_completion(self, mode: str = "error") -> None:
        """Check that all files report a successful completion status.

        Galacticus writes a ``statusCompletion`` attribute to the root of the
        HDF5 file when it finishes.  This method verifies that the attribute
        equals ``"complete"`` for every file in the collection.

        Parameters
        ----------
        mode:
            What to do when an incomplete file is found:

            * ``"error"`` (default) – raise :exc:`RuntimeError`.
            * ``"warn"`` – emit a :class:`UserWarning` and continue.
            * ``"ignore"`` – do nothing.

        Raises
        ------
        ValueError
            If *mode* is not one of the accepted values.
        RuntimeError
            If *mode* is ``"error"`` and at least one file is incomplete.
        """
        if mode not in ("error", "warn", "ignore"):
            raise ValueError(
                f"mode must be 'error', 'warn', or 'ignore'; got {mode!r}"
            )

        incomplete = []
        for path, h in zip(self._files, self._handles):
            status = h.attrs.get("statusCompletion")
            if status != 0:
                incomplete.append((path, status))

        if not incomplete:
            return

        lines = ["The following files have incomplete or missing statusCompletion:"]
        for path, status in incomplete:
            lines.append(f"  {path}: statusCompletion={status!r}")
        msg = "\n".join(lines)

        if mode == "error":
            raise RuntimeError(msg)
        elif mode == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # Output discovery
    # ------------------------------------------------------------------

    @property
    def outputs(self) -> OutputIndex:
        """An :class:`~dendros._outputs.OutputIndex` for this collection.

        The index is scanned lazily on first access and then cached.
        """
        if self._outputs_index is None:
            self._outputs_index = OutputIndex(self)
        return self._outputs_index

    def list_outputs(self, format: str = "astropy"):
        """Return a table of available outputs.

        Scans all ``Output*`` groups inside ``/{output_root}/`` and extracts
        ``outputTime`` and ``outputExpansionFactor`` attributes.  Redshift is
        computed as *z = 1/a − 1*.

        Parameters
        ----------
        format:
            ``"astropy"`` (default) returns an :class:`astropy.table.Table`;
            ``"pandas"`` returns a :class:`pandas.DataFrame`;
            ``"tabulate"`` returns a ``str`` formatted using the ``tabulate`` library.

        Returns
        -------
        astropy.table.Table, pandas.DataFrame, or tabulate-formatted string
        """
        return self.outputs.table(format=format)

    # ------------------------------------------------------------------
    # Properties table
    # ------------------------------------------------------------------

    def list_properties(
        self, output: Union[str, int], format: str = "astropy"
    ):
        """Return a table of datasets available in the ``nodeData`` group.

        Parameters
        ----------
        output:
            Output name (e.g. ``"Output1"``) or 1-based integer index.
        format:
            ``"astropy"`` (default), ``"pandas"``, or ``"tabulate"``.

        Returns
        -------
        astropy.table.Table, pandas.DataFrame, or tabulate-formatted string
        """
        output_name = _resolve_output_name(output)
        path = f"{self._output_root}/{output_name}/nodeData"
        try:
            group = self._primary[path]
        except KeyError:
            raise KeyError(
                f"nodeData group not found at '{path}'. "
                f"Check output_root='{self._output_root}' and "
                f"output='{output_name}'."
            ) from None

        rows: List[dict] = []
        for name in sorted(group.keys()):
            ds = group[name]
            attrs = {k: _decode(v) for k, v in ds.attrs.items()}
            raw_units = attrs.get("unitsInSI")
            try:
                units_val = float(raw_units) if raw_units not in (None, "") else 1.0
            except (TypeError, ValueError):
                units_val = raw_units
            rows.append(
                {
                    "name": name,
                    "dtype": str(ds.dtype),
                    "shape": str(ds.shape),
                    "description": attrs.get("comment", ""),
                    "unitsInSI": units_val,
                }
            )

        return _make_table(rows, format=format, maxcolwidths=[None, None, None, None, 25, None])

    # ------------------------------------------------------------------
    # Reading datasets
    # ------------------------------------------------------------------

    def read(
        self,
        output: Union[str, int],
        datasets: Union[List[str], Dict[str, str]],
        where=None,
    ) -> Dict[str, np.ndarray]:
        """Read one or more datasets from an output group.

        For multi-file collections, arrays from all files are concatenated
        along axis 0 before any selection is applied.

        Parameters
        ----------
        output:
            Output name (e.g. ``"Output1"``) or 1-based integer index.
        datasets:
            Either a list of relative dataset paths under the output group
            (e.g. ``["nodeData/haloMass"]``), in which case the same strings
            are used as dict keys in the return value; or a :class:`dict`
            mapping user-chosen labels to relative paths.
        where:
            ``None`` reads all rows.  A boolean mask array of length
            *N_total* or an integer index array selects a subset.

        Returns
        -------
        dict
            Mapping from dataset name / label to :class:`numpy.ndarray`.

        Notes
        -----
        The ``unitsInSI`` attribute is preserved in the raw array values
        but not yet applied.  Future versions will optionally return
        :class:`astropy.units.Quantity` objects.
        """
        output_name = _resolve_output_name(output)
        output_path = f"{self._output_root}/{output_name}"

        if isinstance(datasets, dict):
            datasets_map: Dict[str, str] = datasets
        else:
            datasets_map = {ds: ds for ds in datasets}

        result: Dict[str, np.ndarray] = {}
        for label, rel_path in datasets_map.items():
            full_path = output_path + "/" + rel_path.lstrip("/")
            arrays: List[np.ndarray] = []
            for h in self._handles:
                try:
                    arrays.append(h[full_path][()])
                except KeyError:
                    raise KeyError(
                        f"Dataset '{full_path}' not found in '{h.filename}'"
                    ) from None
            arr = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
            if where is not None:
                arr = arr[np.asarray(where)]
            result[label] = arr

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for h in self._handles:
            try:
                h.close()
            except Exception:
                pass
        self._handles = []

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------


def _make_table(rows: List[dict], format: str, **kwargs):
    """Convert a list of row dicts to the requested table format."""
    if format == "astropy":
        from astropy.table import Table

        if not rows:
            return Table()
        keys = list(rows[0].keys())
        data = {k: [r[k] for r in rows] for k in keys}
        return Table(data)
    elif format == "pandas":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is not installed. "
                "Install it with: pip install 'dendros[pandas]'"
            ) from exc
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    elif format == "tabulate":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is not installed. "
                "Install it with: pip install 'dendros[pandas]'"
            ) from exc
        try:
            from tabulate import tabulate
        except ImportError as exc:
            raise ImportError(
                "tabulate is not installed. "
                "Install it with: pip install 'dendros[tabulate]'"
            ) from exc
        if not rows:
            return ""
        return tabulate(pd.DataFrame(rows), headers=list(rows[0].keys()), **kwargs)
    else:
        raise ValueError(f"format must be 'astropy', 'pandas', or 'tabulate'; got {format!r}")


# ---------------------------------------------------------------------------
# open_outputs
# ---------------------------------------------------------------------------


def open_outputs(
    path: Union[str, "Path", List[Union[str, "Path"]]],
    output_root: str = "Outputs",
) -> Collection:
    """Open a Galacticus output collection.

    Parameters
    ----------
    path:
        One of:

        * A single filename – e.g. ``"galacticus.hdf5"``.  If sibling
          MPI-rank files (``*_MPI:????``) exist they are included
          automatically.
        * A glob string – e.g. ``"run*/galacticus*.hdf5"``.
        * An explicit list of filenames.

    output_root:
        Top-level HDF5 group containing the ``Output*`` groups.
        Defaults to ``"Outputs"``.  Pass ``"Lightcone"`` for lightcone runs
        or any other custom group name as needed.

    Returns
    -------
    Collection

    Raises
    ------
    FileNotFoundError
        If no files are found matching *path*.
    TypeError
        If *path* is not a ``str``, :class:`pathlib.Path`, or ``list``.

    Examples
    --------
    Open a single file::

        c = open_outputs("galacticus.hdf5")

    Auto-detect MPI-split files (given any one rank's file)::

        c = open_outputs("galacticus_MPI:0000.hdf5")

    Open via glob::

        c = open_outputs("run001/galacticus*.hdf5")

    Open an explicit list::

        c = open_outputs(["file_a.hdf5", "file_b.hdf5"])

    Lightcone mode::

        c = open_outputs("lightcone.hdf5", output_root="Lightcone")
    """
    if isinstance(path, list):
        files = [str(p) for p in path]
    elif isinstance(path, (str, Path)):
        path = str(path)
        expanded = sorted(_glob.glob(path))
        if not expanded:
            raise FileNotFoundError(f"No files found matching: {path!r}")
        if len(expanded) == 1:
            files = _auto_detect_mpi(expanded[0])
        else:
            files = expanded
    else:
        raise TypeError(
            f"path must be str, Path, or list; got {type(path).__name__!r}"
        )

    if not files:
        raise FileNotFoundError("No HDF5 files found.")

    return Collection(files, output_root=output_root)


def _auto_detect_mpi(path: str) -> List[str]:
    """Return sorted MPI-rank peers if they exist; otherwise return ``[path]``."""
    p = Path(path)
    stem = p.stem
    m = _MPI_SUFFIX.match(stem)
    base = m.group(1) if m else stem
    peers = sorted(p.parent.glob(f"{base}:MPI????{p.suffix}"))
    if peers:
        return [str(x) for x in peers]
    return [path]
