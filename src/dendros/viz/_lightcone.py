"""Mock observed images of a Galacticus lightcone.

Every galaxy crossing the lightcone is rendered from the model's own description of it
(density grids and observed-frame luminosities, as for :func:`~dendros.viz.galaxy_scene`),
at its position on the sky, its angular size (from the angular diameter distance at its
redshift), and its orientation (its disk's axis along its spin), with dust absorbing at
the rest-frame wavelength. Fluxes follow from the luminosity distance, so the image is
in nJy per pixel; a Gaussian point-spread function and noise may be added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._galaxy import GalaxyScene, GasComponent, StellarComponent
from ._profiles import density_tables
from ._render import DustModel, render_galaxy

#: The AB zero point [nJy].
_AB_ZERO_NJY = 3631.0e9
_ARCSEC = np.pi / 180.0 / 3600.0


@dataclass
class LightconeGalaxy:
    """One galaxy in a lightcone, ready to render."""

    scene: GalaxyScene
    #: Observed redshift (including peculiar velocity).
    redshift: float
    #: Cosmological redshift (for distances).
    redshift_cosmological: float
    #: Position on the sky [arcsec], along the field's two axes.
    sky: np.ndarray
    #: Unit vector along the galaxy's spin, in the camera frame (x, y on the sky; z towards the observer).
    axis: np.ndarray


@dataclass
class LightconeField:
    """The galaxies of a lightcone and the geometry needed to image them."""

    galaxies: List[LightconeGalaxy]
    #: Angular size of the (square) field [arcsec].
    size: float
    #: Hubble constant [km/s/Mpc] and density parameters, for distances.
    cosmology: Dict[str, float] = field(default_factory=dict)

    def distances(self, redshift):
        """Angular diameter and luminosity distances [Mpc] at ``redshift``."""
        from astropy.cosmology import LambdaCDM

        c = self.cosmology
        cosmology = LambdaCDM(
            H0=c["HubbleConstant"],
            Om0=c["OmegaMatter"],
            Ode0=c["OmegaDarkEnergy"],
            Ob0=c.get("OmegaBaryon"),
        )
        return (
            cosmology.angular_diameter_distance(redshift).value,
            cosmology.luminosity_distance(redshift).value,
        )


def _parameter(group, path) -> np.ndarray:
    """Read a numeric parameter (scalar or list) from the model's ``Parameters`` group, e.g.
    ``geometryLightcone/angularSize``. Values are stored either as numbers or as text.
    """
    *parents, name = path.split("/")
    for p in parents:
        group = group[p]
    value = group.attrs[name]
    if isinstance(value, (bytes, np.bytes_, str)):
        text = value.decode() if isinstance(value, (bytes, np.bytes_)) else value
        return np.array(
            [float(x) for x in text.replace("[", " ").replace("]", " ").split()]
        )
    return np.atleast_1d(np.asarray(value, dtype=float))


def lightcone_field(
    c,
    bands: Sequence[str] = (
        "JWST_NIRCAM_f115w",
        "JWST_NIRCAM_f200w",
        "JWST_NIRCAM_f444w",
    ),
    output: str = "Output1",
    total: str = "inoue2014",
    young: Optional[str] = "sequence",
    mass_minimum: float = 0.0,
    orientation: str = "auto",
    seed: int = 1,
) -> LightconeField:
    """Read the galaxies of a lightcone, ready to image.

    ``c`` is a collection opened with ``output_root="Lightcone"``. The model must write,
    at lightcone crossings, the ``lightcone`` extractor's positions and observed redshifts,
    the ``spin`` extractor's angular momentum vectors, the ``densityProfile`` grids (as for
    :func:`~dendros.viz.galaxy_scene`), and observed-frame luminosities from the
    ``luminosityStellarFromStarFormationHistory`` extractor for each component in each band:
    those of all stars with spectra postprocessor ``total``, and those of young stars with
    ``young`` (``None`` if absent).  Galaxies below ``mass_minimum`` in stellar mass are
    skipped.

    Disks are oriented with their axis along the halo spin vector (``orientation="spin"``,
    taken to be in the simulation's frame), or at random, isotropically (``"random"``,
    reproducibly, from ``seed`` and each galaxy's node index). Galacticus predicts no disk
    orientation itself; spin vectors carry one only if the model presets the halos' 3-D
    angular momenta from the merger trees, and with ``"auto"`` random orientations are used
    when the spin vectors carry no direction (all along x).
    """
    root = c["Parameters"]
    units = [_parameter(root, f"geometryLightcone/unitVector{i}") for i in (1, 2, 3)]
    size = float(_parameter(root, "geometryLightcone/angularSize")[0]) * 3600.0
    cosmology = {
        name: float(_parameter(root, f"cosmologyParameters/{name}")[0])
        for name in ("HubbleConstant", "OmegaMatter", "OmegaDarkEnergy", "OmegaBaryon")
    }
    group = c[f"{c.output_root}/{output}/nodeData"]
    available = set(group.keys())

    def column(name, default=None):
        if name not in available:
            if default is None:
                raise KeyError(f"lightcone output lacks '{name}'")
            return np.full(
                group["lightconeRedshiftObserved"].shape[0], default, dtype=float
            )
        return np.asarray(group[name][()])

    densities, radii = column("densityProfile"), column("densityProfileRadius")
    descriptions = np.asarray(group["densityProfileColumns"][()])
    angles = np.asarray(group["densityProfileColumnValues"][()])
    position = np.stack([column(f"lightconePosition{a}") for a in "XYZ"], axis=1)
    spin = np.stack([column(f"spinAngularMomentumVector{a}") for a in "XYZ"], axis=1)
    redshift_observed = column("lightconeRedshiftObserved")
    redshift = column("lightconeRedshiftCosmological")
    mass_stellar = column("diskMassStellar", 0.0) + column("spheroidMassStellar", 0.0)
    components = sorted(
        {
            d.decode().split(":")[1] if isinstance(d, bytes) else str(d).split(":")[1]
            for d in descriptions
        }
    )
    luminosity = {}
    for component in components:
        for band in bands:
            name = f"{component}StellarSFH:{total}:{band}:observed"
            if name in available:
                luminosity[(component, band, False)] = column(name)
            if (
                young is not None
                and f"{component}StellarSFH:{young}:{band}:observed" in available
            ):
                luminosity[(component, band, True)] = column(
                    f"{component}StellarSFH:{young}:{band}:observed"
                )
    scalars = {
        name: column(name, 0.0)
        for name in (
            "diskMassStellar",
            "spheroidMassStellar",
            "diskMassGas",
            "spheroidMassGas",
            "diskAbundancesGasMetals",
            "spheroidAbundancesGasMetals",
            "diskRadius",
            "spheroidRadius",
            "radiusHalfMassStellar",
            "nodeIndex",
        )
    }
    # Positions are in the lightcone's frame (x along the line of sight, y and z along the field's axes). The camera frame
    # has x and y on the sky (along the field's z and y axes, so that x, y and the direction towards the observer, -x, form
    # a right-handed set).
    lightcone_to_camera = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    box_to_lightcone = np.array(units)
    if orientation == "auto":
        directionless = np.all(spin[:, 1:] == 0.0)
        orientation = "random" if directionless else "spin"
    if orientation not in ("spin", "random"):
        raise ValueError("orientation must be 'auto', 'spin' or 'random'")
    galaxies = []
    for i in np.nonzero(mass_stellar >= mass_minimum)[0]:
        tables = density_tables(densities[i], radii[i], descriptions, angles)
        stars, gas = [], []
        for (component, mass_type), table in tables.items():
            if mass_type == "stellar":
                lum = {
                    b: float(luminosity[(component, b, False)][i])
                    for b in bands
                    if (component, b, False) in luminosity
                }
                lum_young = {
                    b: float(luminosity[(component, b, True)][i])
                    for b in bands
                    if (component, b, True) in luminosity
                }
                if lum:
                    stars.append(
                        StellarComponent(
                            component,
                            table,
                            float(scalars[f"{component}MassStellar"][i]),
                            lum,
                            lum_young,
                        )
                    )
            elif mass_type == "gaseous":
                mass = float(scalars[f"{component}MassGas"][i])
                metals = float(scalars[f"{component}AbundancesGasMetals"][i])
                gas.append(
                    GasComponent(
                        component, table, mass, metals / mass if mass > 0.0 else 0.0
                    )
                )
        scene = GalaxyScene(
            stars,
            gas,
            list(bands),
            int(scalars["nodeIndex"][i]),
            -1,
            float(scalars["diskRadius"][i]),
            float(scalars["spheroidRadius"][i]),
            float(scalars["radiusHalfMassStellar"][i]),
        )
        sky = np.array([position[i, 2], position[i, 1]]) / position[i, 0] / _ARCSEC
        if orientation == "spin" and np.linalg.norm(spin[i]) > 0.0:
            axis = (
                lightcone_to_camera
                @ box_to_lightcone
                @ (spin[i] / np.linalg.norm(spin[i]))
            )
        else:
            rng = np.random.default_rng(
                [seed, int(scalars["nodeIndex"][i]) & 0x7FFFFFFF, int(i)]
            )
            v = rng.normal(size=3)
            axis = v / np.linalg.norm(v)
        galaxies.append(
            LightconeGalaxy(
                scene, float(redshift_observed[i]), float(redshift[i]), sky, axis
            )
        )
    return LightconeField(galaxies, size, cosmology)


def _rotation_to_galaxy(axis: np.ndarray) -> np.ndarray:
    """The matrix taking camera coordinates to a galaxy frame whose z axis is ``axis`` (in camera coordinates)."""
    z = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def render_lightcone(
    field: LightconeField,
    bands: Optional[Sequence[str]] = None,
    pixel_scale: float = 0.04,
    psf_fwhm: Optional[Dict[str, float]] = None,
    oversample: int = 3,
    extent: float = 10.0,
    taper: float = 0.25,
    stamp_maximum: int = 900,
    dust: Optional[DustModel] = None,
    texture: bool = False,
    young_scale_height: Optional[Dict[str, float]] = None,
    n_steps: int = 256,
    threads: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, np.ndarray]:
    """Image a lightcone: one array per band, in nJy per pixel.

    Each galaxy is rendered on its own stamp, ``extent`` stellar half-mass radii in
    half-width (and at least a few pixels), at up to ``oversample`` times the pixel
    resolution (less for stamps wider than ``stamp_maximum`` fine pixels), then binned and
    added into the field at its position.  The outer ``taper`` fraction of each stamp's
    radius is faded out smoothly, so that no stamp edge shows (at the cost of the little
    light beyond).  Its luminosity per unit area is converted to flux with the luminosity
    distance (including the factor 1/(1+z) of Galacticus' observed-frame luminosities),
    and its angular size follows from the angular diameter distance.  ``psf_fwhm``
    [arcsec] per band convolves each band with a Gaussian point-spread function.
    ``texture`` adds the (non-model) spiral texture to disks.  Galaxies are rendered in
    ``threads`` parallel threads (default: up to 8).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    from scipy.ndimage import gaussian_filter

    from ._texture import SpiralTexture

    bands = list(bands) if bands is not None else list(field.galaxies[0].scene.bands)
    npix = int(round(field.size / pixel_scale))
    images = {b: np.zeros((npix, npix), dtype=np.float64) for b in bands}
    dust = dust or DustModel()

    def stamp_of(galaxy):
        """Render one galaxy: returns (x0, y0, {band: binned flux [nJy]}) or None."""
        scene = galaxy.scene
        if not scene.stars or galaxy.redshift_cosmological <= 0.0:
            return None
        distance_angular, distance_luminosity = field.distances(
            galaxy.redshift_cosmological
        )
        # Flux [nJy] from a luminosity of one Galacticus unit (AB magnitude zero at 10 pc), in an observed-frame band.
        modulus = 5.0 * np.log10(distance_luminosity * 1.0e6 / 10.0) - 2.5 * np.log10(
            1.0 + galaxy.redshift
        )
        flux_unit = _AB_ZERO_NJY * 10.0 ** (-0.4 * modulus)
        mpc_per_arcsec = distance_angular * _ARCSEC
        half_angle = min(
            max(scene.extent(extent) / mpc_per_arcsec, 4.0 * pixel_scale),
            0.25 * field.size,
        )
        stamp = 2 * int(np.ceil(half_angle / pixel_scale)) + 1
        half_angle = 0.5 * stamp * pixel_scale
        factor = int(np.clip(stamp_maximum // stamp, 1, oversample))
        fine = stamp * factor
        tex = (
            SpiralTexture(
                radius_scale=scene.radius_disk, seed=int(scene.node_index) % 100000
            )
            if texture and scene.radius_disk > 0.0
            else None
        )
        rendered = render_galaxy(
            scene,
            bands=bands,
            npix=fine,
            half_width=half_angle * mpc_per_arcsec,
            n_steps=n_steps,
            dust=dust,
            texture=tex,
            rotation=_rotation_to_galaxy(galaxy.axis),
            redshift=galaxy.redshift_cosmological,
            young_scale_height=young_scale_height,
            threads=1,
        )
        # A circular window, fading out over the outer part of the stamp.
        u = (np.arange(fine) + 0.5) / fine * 2.0 - 1.0
        r = np.hypot(*np.meshgrid(u, u))
        window = np.clip((1.0 - r) / max(taper, 1.0e-6), 0.0, 1.0)
        window = window * window * (3.0 - 2.0 * window)
        # Surface brightness [L/pc^2] -> flux per fine pixel [nJy], then binned to the field's pixels.
        area_pc2 = (2.0 * half_angle * mpc_per_arcsec * 1.0e6 / fine) ** 2
        binned = {
            b: (rendered[b][::-1] * window * area_pc2 * flux_unit)
            .reshape(stamp, factor, stamp, factor)
            .sum(axis=(1, 3))
            for b in bands
        }  # row 0 at the top of the sky
        cx = int(round(galaxy.sky[0] / pixel_scale + npix / 2.0))
        cy = int(round(npix / 2.0 - galaxy.sky[1] / pixel_scale))
        return cx - stamp // 2, cy - stamp // 2, binned

    threads = threads or min(8, os.cpu_count() or 1)
    done = 0
    # Render the largest galaxies first, so the threads finish together.
    order = sorted(
        range(len(field.galaxies)),
        key=lambda i: -field.galaxies[i].scene.extent(1.0)
        / max(
            field.distances(max(field.galaxies[i].redshift_cosmological, 1.0e-3))[0],
            1.0e-30,
        ),
    )
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for result in pool.map(stamp_of, [field.galaxies[i] for i in order]):
            done += 1
            if progress and done % 200 == 0:
                print(f"rendered {done} of {len(field.galaxies)} galaxies", flush=True)
            if result is None:
                continue
            x0, y0, binned = result
            stamp = next(iter(binned.values())).shape[0]
            ys, xs = slice(max(y0, 0), min(y0 + stamp, npix)), slice(
                max(x0, 0), min(x0 + stamp, npix)
            )
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            for b in bands:
                images[b][ys, xs] += binned[b][
                    ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
                ]
    if psf_fwhm:
        for b in bands:
            if psf_fwhm.get(b):
                sigma = psf_fwhm[b] / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                images[b] = gaussian_filter(images[b], sigma)
    return images


def add_noise(
    images: Dict[str, np.ndarray],
    depth: Dict[str, float],
    pixel_scale: float,
    aperture: float = 0.2,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to images [nJy per pixel] for a 5σ point-source depth ``depth`` [AB mag] per band, measured in
    a circular aperture of radius ``aperture`` [arcsec]."""
    rng = np.random.default_rng(seed)
    pixels = np.pi * (aperture / pixel_scale) ** 2
    out = {}
    for b, image in images.items():
        sigma_aperture = _AB_ZERO_NJY * 10.0 ** (-0.4 * depth[b]) / 5.0
        out[b] = image + rng.normal(0.0, sigma_aperture / np.sqrt(pixels), image.shape)
    return out
