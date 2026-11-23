"""A single Galacticus galaxy assembled into a renderable scene.

A :class:`GalaxyScene` holds, for each stellar component, its tabulated density
and its luminosity per unit stellar mass in each band (split into light from
recently formed stars and from older stars when the model provides both), and
for the gas its tabulated density and metallicity, from which the dust follows.

Light is assumed to follow the stellar mass of its component: Galacticus does not
resolve stellar populations within a component, so each component has a single
color.  Color gradients in the images come from the mix of components (blue disk,
red bulge), from dust, and from any (optional, non-model) texture applied to the
young light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._profiles import DensityTable, density_tables

#: Effective wavelengths of the SDSS filters [Å], used to scale the dust opacity.
WAVELENGTH_EFFECTIVE = {
    "SDSS_u": 3551.0,
    "SDSS_g": 4686.0,
    "SDSS_r": 6166.0,
    "SDSS_i": 7480.0,
    "SDSS_z": 8932.0,
    # Pivot wavelengths of HST, JWST and Euclid filters [Å].
    "ACSWFC_f435w": 4329.0,
    "ACSWFC_f606w": 5922.0,
    "ACSWFC_f814w": 8045.0,
    "ACSWFC_f850lp": 9033.0,
    "JWST_NIRCAM_f090w": 9022.0,
    "JWST_NIRCAM_f115w": 11543.0,
    "JWST_NIRCAM_f150w": 15007.0,
    "JWST_NIRCAM_f200w": 19886.0,
    "JWST_NIRCAM_f277w": 27617.0,
    "JWST_NIRCAM_f356w": 35684.0,
    "JWST_NIRCAM_f444w": 44043.0,
    "Euclid_VIS": 7150.0,
    "Euclid_Y": 10800.0,
    "Euclid_J": 13700.0,
    "Euclid_H": 17700.0,
}


@dataclass
class StellarComponent:
    name: str
    density: DensityTable
    mass: float
    #: Luminosity of all stars per band [Galacticus luminosity units; AB magnitude 0 at 10 pc].
    luminosity: Dict[str, float]
    #: Luminosity of recently formed stars per band (same units), if the model provides it.
    luminosity_young: Dict[str, float] = field(default_factory=dict)

    def light_to_mass(self, band: str, young: Optional[bool] = None) -> float:
        """Luminosity per unit stellar mass in ``band``.

        ``young=None`` gives all stars, ``True`` only recently formed stars, ``False``
        only the older ones.
        """
        if self.mass <= 0.0:
            return 0.0
        total = self.luminosity.get(band, 0.0)
        if young is None:
            return total / self.mass
        recent = self.luminosity_young.get(band, 0.0)
        return (recent if young else total - recent) / self.mass


@dataclass
class GasComponent:
    name: str
    density: DensityTable
    mass: float
    metallicity: float


@dataclass
class GalaxyScene:
    """Everything needed to render one galaxy, in the galaxy's own frame (z along the disk axis)."""

    stars: List[StellarComponent]
    gas: List[GasComponent]
    bands: List[str]
    node_index: int
    tree_index: int
    #: Scale radii of the disk and spheroid [Mpc] (zero if absent).
    radius_disk: float
    radius_spheroid: float
    radius_half_mass: float

    def extent(self, factor: float = 4.0) -> float:
        """A sensible half-width for an image [Mpc]: ``factor`` stellar half-mass radii."""
        r = self.radius_half_mass
        if not r > 0.0:
            r = max(self.radius_disk, self.radius_spheroid)
        return factor * r


def _read(c, output, name, where):
    return np.asarray(
        c.read(output, [f"nodeData/{name}"], as_quantity=False, where=where)[
            f"nodeData/{name}"
        ]
    )


def _luminosity_names(available: Sequence[str], component: str, band: str):
    """Return (all-stars, recent-stars) dataset names for ``component`` in ``band``."""
    prefix = f"{component}LuminositiesStellar:{band}:"
    matches = [n for n in available if n.startswith(prefix)]
    recent = [n for n in matches if ":recent" in n]
    total = [n for n in matches if ":recent" not in n]
    return (total[0] if total else None), (recent[0] if recent else None)


def galaxy_scene(
    c, output, index: int, bands: Sequence[str] = ("SDSS_g", "SDSS_r", "SDSS_i")
) -> GalaxyScene:
    """Assemble the :class:`GalaxyScene` for the galaxy at row ``index`` of ``output``.

    The model must have run the ``densityProfile`` extractor with ``includeRadii=true``
    and radius specifiers covering ``disk:stellar``, ``disk:gaseous`` and
    ``spheroid:stellar`` (other components are used if present), and must output
    stellar luminosities per component in ``bands``.
    """
    where = np.array([index])
    group = c[f"Outputs/{_output_name(c, output)}/nodeData"]
    available = list(group.keys())
    for required in ("densityProfile", "densityProfileRadius", "densityProfileColumns"):
        if required not in available:
            raise KeyError(
                f"output lacks '{required}': run the densityProfile extractor with includeRadii=true"
            )
    densities = _read(c, output, "densityProfile", where)[0]
    radii = _read(c, output, "densityProfileRadius", where)[0]
    descriptions = np.asarray(group["densityProfileColumns"][()])
    if "densityProfileColumnValues" in available:
        angles = np.asarray(group["densityProfileColumnValues"][()])
    else:
        angles = np.full(densities.shape, 90.0)  # equatorial plane only
    tables = density_tables(densities, radii, descriptions, angles)

    def scalar(name, default=0.0):
        return float(_read(c, output, name, where)[0]) if name in available else default

    stars, gas = [], []
    for (component, mass_type), table in tables.items():
        if mass_type == "stellar":
            lum, lum_young = {}, {}
            for band in bands:
                total, recent = _luminosity_names(available, component, band)
                if total is None:
                    raise KeyError(f"output lacks {component} luminosities in {band}")
                lum[band] = scalar(total)
                if recent is not None:
                    lum_young[band] = scalar(recent)
            stars.append(
                StellarComponent(
                    component, table, scalar(f"{component}MassStellar"), lum, lum_young
                )
            )
        elif mass_type == "gaseous":
            mass = scalar(f"{component}MassGas")
            metals = scalar(f"{component}AbundancesGasMetals")
            gas.append(
                GasComponent(
                    component, table, mass, metals / mass if mass > 0.0 else 0.0
                )
            )
    return GalaxyScene(
        stars=stars,
        gas=gas,
        bands=list(bands),
        node_index=int(scalar("nodeIndex", -1)),
        tree_index=int(scalar("mergerTreeIndex", -1)),
        radius_disk=scalar("diskRadius"),
        radius_spheroid=scalar("spheroidRadius"),
        radius_half_mass=scalar("radiusHalfMassStellar"),
    )


def _output_name(c, output) -> str:
    if isinstance(output, str):
        return output
    return f"Output{int(output)}"
