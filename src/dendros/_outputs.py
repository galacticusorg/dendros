"""Output metadata index for Galacticus HDF5 collections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

if TYPE_CHECKING:
    from ._collection import Collection


@dataclass(frozen=True)
class OutputMeta:
    """Metadata for a single Galacticus output snapshot.

    Attributes
    ----------
    name:
        Name of the output group (e.g. ``"Output1"``).
    path:
        Full HDF5 path to the output group.
    index:
        1-based sequential index (order of discovery, sorted numerically).
    time:
        Output time.  Units are determined by the Galacticus configuration
        (typically Gyr), or ``None`` if the attribute is absent.
    scale_factor:
        Expansion factor *a*, or ``None`` if absent.
    redshift:
        Redshift *z = 1/a − 1*, computed from ``scale_factor``, or ``None``.
    """

    name: str
    path: str
    index: int
    time: Optional[float]
    scale_factor: Optional[float]
    redshift: Optional[float]


class OutputIndex:
    """Index of all ``Output*`` groups found in a :class:`~dendros.Collection`.

    Instances are obtained via :attr:`~dendros.Collection.outputs`.

    Parameters
    ----------
    collection:
        The parent :class:`~dendros.Collection`.
    """

    def __init__(self, collection: "Collection") -> None:
        self._collection = collection
        self._outputs: List[OutputMeta] = []
        self._scan()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        """Populate ``self._outputs`` by scanning the output root group."""
        h = self._collection._primary
        root = self._collection.output_root
        try:
            root_group = h[root]
        except KeyError:
            raise KeyError(
                f"Output root group '{root}' not found in the HDF5 file. "
                f"Pass a different output_root= when calling open_outputs() "
                f"(e.g. output_root='Lightcone' for lightcone runs)."
            ) from None

        output_names = sorted(
            [k for k in root_group.keys() if k.startswith("Output")],
            key=_output_sort_key,
        )
        for i, name in enumerate(output_names):
            group = root_group[name]
            attrs = dict(group.attrs)
            time = _float_or_none(attrs.get("outputTime"))
            a = _float_or_none(attrs.get("outputExpansionFactor"))
            z = (1.0 / a - 1.0) if (a is not None and a > 0) else None
            self._outputs.append(
                OutputMeta(
                    name=name,
                    path=f"/{root}/{name}",
                    index=i + 1,
                    time=time,
                    scale_factor=a,
                    redshift=z,
                )
            )

    # ------------------------------------------------------------------
    # Sequence-like interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._outputs)

    def __iter__(self) -> Iterator[OutputMeta]:
        return iter(self._outputs)

    def __getitem__(self, key: Union[int, str]) -> OutputMeta:
        """Look up an output by 0-based integer position or by name.

        Parameters
        ----------
        key:
            An integer index (0-based) or a group name such as ``"Output3"``.
        """
        if isinstance(key, int):
            return self._outputs[key]
        for o in self._outputs:
            if o.name == key:
                return o
        raise KeyError(f"No output named {key!r}")

    # ------------------------------------------------------------------
    # Table output
    # ------------------------------------------------------------------

    def table(self, format: str = "astropy"):
        """Return a table of output metadata.

        Parameters
        ----------
        format:
            ``"astropy"`` (default) returns an :class:`astropy.table.Table`;
            ``"pandas"`` returns a :class:`pandas.DataFrame`.

        Returns
        -------
        astropy.table.Table or pandas.DataFrame
        """
        rows = [
            {
                "index": o.index,
                "name": o.name,
                "time": o.time,
                "scale_factor": o.scale_factor,
                "redshift": o.redshift,
            }
            for o in self._outputs
        ]
        from ._collection import _make_table

        return _make_table(rows, format=format)

    def __repr__(self) -> str:
        return f"<OutputIndex n={len(self._outputs)}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_sort_key(name: str) -> int:
    """Sort ``Output*`` names numerically by their trailing integer."""
    suffix = name[len("Output"):]
    return int(suffix) if suffix.isdigit() else 0


def _float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
