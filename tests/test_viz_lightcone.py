"""Tests for mock observed lightcone images in dendros.viz."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("astropy")

from dendros.viz import (  # noqa: E402
    DustModel,
    GalaxyScene,
    LightconeField,
    LightconeGalaxy,
    StellarComponent,
    render_lightcone,
)
from dendros.viz._lightcone import _rotation_to_galaxy  # noqa: E402

from .test_viz import RS, _disk_table  # noqa: E402

COSMOLOGY = dict(
    HubbleConstant=73.0, OmegaMatter=0.25, OmegaDarkEnergy=0.75, OmegaBaryon=0.045
)
BAND = "JWST_NIRCAM_f200w"
LUMINOSITY = 1.0e10


def _field(redshift=1.0, sky=(0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    disk = StellarComponent("disk", _disk_table(), 5.0e10, {BAND: LUMINOSITY})
    scene = GalaxyScene([disk], [], [BAND], 1, 1, RS, 0.0, 1.68 * RS)
    galaxy = LightconeGalaxy(
        scene, redshift, redshift, np.array(sky), np.array(axis, dtype=float)
    )
    return LightconeField([galaxy], 20.0, COSMOLOGY)


def test_rotation_to_galaxy():
    for axis in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.3, -0.5, 0.8]):
        m = _rotation_to_galaxy(np.array(axis))
        assert np.allclose(m @ m.T, np.eye(3))
        assert np.allclose(m @ (np.array(axis) / np.linalg.norm(axis)), [0.0, 0.0, 1.0])


def test_total_flux():
    """The flux in the image is that of the galaxy's luminosity at its distance (AB, with the (1+z) term)."""
    from astropy.cosmology import LambdaCDM

    redshift = 1.0
    field = _field(redshift)
    image = render_lightcone(
        field,
        [BAND],
        pixel_scale=0.05,
        oversample=3,
        extent=11.0 / 1.68,
        dust=DustModel(enabled=False),
        n_steps=512,
    )[BAND]
    cosmology = LambdaCDM(H0=73.0, Om0=0.25, Ode0=0.75, Ob0=0.045)
    distance = cosmology.luminosity_distance(redshift).value * 1.0e6  # pc
    magnitude = (
        -2.5 * np.log10(LUMINOSITY)
        + 5.0 * np.log10(distance / 10.0)
        - 2.5 * np.log10(1.0 + redshift)
    )
    expected = 3631.0e9 * 10.0 ** (-0.4 * magnitude)
    # The table holds the disk out to 10 scale radii, so the light within it is 1 - 11 e^-10 of the total.
    assert image.sum() == pytest.approx(
        expected * (1.0 - 11.0 * np.exp(-10.0)), rel=0.01
    )


def test_position_and_orientation():
    # Face-on (axis along the line of sight): a round image, centered where placed.
    field = _field(sky=(2.0, -3.0))
    image = render_lightcone(
        field, [BAND], pixel_scale=0.05, dust=DustModel(enabled=False)
    )[BAND]
    y, x = np.unravel_index(np.argmax(image), image.shape)
    assert (x - 200) * 0.05 == pytest.approx(2.0, abs=0.06) and (
        200 - y
    ) * 0.05 == pytest.approx(-3.0, abs=0.06)
    face = image[y - 15 : y + 16, x - 15 : x + 16]
    assert np.allclose(face, face.T, rtol=0.05, atol=1.0e-6 * face.max())
    # Edge-on, with the axis along the sky's y direction: elongated along x.
    edge = render_lightcone(
        _field(axis=(0.0, 1.0, 0.0)),
        [BAND],
        pixel_scale=0.05,
        dust=DustModel(enabled=False),
    )[BAND]
    y, x = np.unravel_index(np.argmax(edge), edge.shape)
    assert edge[y, x + 4] > 5.0 * edge[y + 4, x]


def _write_lightcone(path, spin):
    """A lightcone file with two galaxies (disks only), in the layout Galacticus writes."""
    import h5py

    from .test_viz import FACTORS, OFFSETS, _disk_density

    angles = np.sort(90.0 - OFFSETS)[::-1]
    descriptions, values, radii, rho = [], [], [], []
    for mass_type in ("stellar", "gaseous"):
        for a in angles:
            for f in FACTORS:
                descriptions.append(
                    f"diskRadius:disk:{mass_type}:{f:.5g} θ={a:9.4f}°".encode()
                )
                values.append(a)
                radii.append(f * RS)
                rho.append(_disk_density(f * RS, np.deg2rad(a)))
    n = 2
    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = 0
        p = f.create_group("Parameters")
        g = p.create_group("geometryLightcone")
        g.attrs["angularSize"] = 0.01
        for i, u in enumerate(np.eye(3)):
            g.attrs[f"unitVector{i + 1}"] = u
        c = p.create_group("cosmologyParameters")
        for name, value in COSMOLOGY.items():
            c.attrs[name] = value
        out = f.create_group("Lightcone/Output1")
        out.attrs["outputTime"] = 13.0
        out.attrs["outputExpansionFactor"] = 1.0
        d = out.create_group("nodeData")
        d["densityProfile"] = np.tile(np.array(rho), (n, 1))
        d["densityProfileRadius"] = np.tile(np.array(radii), (n, 1))
        d["densityProfileColumns"] = np.array(descriptions)
        d["densityProfileColumnValues"] = np.array(values)
        # Galaxy 0 lies 10" along the field's y axis (lightcone z), galaxy 1 at -5" along x (lightcone y = +5").
        distance = 3000.0
        d["lightconePositionX"] = np.array([distance, distance])
        d["lightconePositionY"] = (
            np.array([0.0, 5.0]) * distance * np.pi / 180.0 / 3600.0
        )
        d["lightconePositionZ"] = (
            np.array([10.0, 0.0]) * distance * np.pi / 180.0 / 3600.0
        )
        for a, s in zip("XYZ", np.asarray(spin).T):
            d[f"spinAngularMomentumVector{a}"] = s
        d["lightconeRedshiftObserved"] = np.array([1.0, 1.01])
        d["lightconeRedshiftCosmological"] = np.array([1.0, 1.0])
        d["diskMassStellar"] = np.array([5.0e10, 5.0e10])
        d["diskMassGas"] = np.array([1.0e10, 1.0e10])
        d["diskAbundancesGasMetals"] = np.array([2.0e8, 2.0e8])
        d["diskRadius"] = np.array([RS, RS])
        d["radiusHalfMassStellar"] = np.array([1.68 * RS, 1.68 * RS])
        d["nodeIndex"] = np.array([11, 12])
        d[f"diskStellarSFH:inoue2014:{BAND}:observed"] = np.array([1.0e10, 2.0e10])
        d[f"diskStellarSFH:sequence:{BAND}:observed"] = np.array([1.0e9, 2.0e9])


def test_lightcone_field(tmp_path):
    from dendros import open_outputs
    from dendros.viz import lightcone_field

    path = tmp_path / "lightcone.hdf5"
    _write_lightcone(
        path, [[1.0e9, 0.0, 0.0], [2.0e9, 0.0, 0.0]]
    )  # spins without direction
    with open_outputs(str(path), output_root="Lightcone") as c:
        field = lightcone_field(c, bands=(BAND,))
    assert field.size == pytest.approx(36.0)
    assert field.cosmology["HubbleConstant"] == pytest.approx(73.0)
    a, b = field.galaxies
    # Sky x runs along the lightcone's z axis, sky y along its y axis.
    assert np.allclose(a.sky, [10.0, 0.0], atol=1.0e-6) and np.allclose(
        b.sky, [0.0, 5.0], atol=1.0e-6
    )
    assert a.redshift == pytest.approx(1.0) and b.redshift == pytest.approx(1.01)
    disk = b.scene.stars[0]
    assert disk.luminosity[BAND] == pytest.approx(2.0e10) and disk.luminosity_young[
        BAND
    ] == pytest.approx(2.0e9)
    assert b.scene.gas[0].metallicity == pytest.approx(0.02)
    # Spins without direction: random (but reproducible) orientations.
    assert not np.allclose(a.axis, b.axis) and np.linalg.norm(a.axis) == pytest.approx(
        1.0
    )
    with open_outputs(str(path), output_root="Lightcone") as c:
        again = lightcone_field(c, bands=(BAND,))
    assert np.allclose(again.galaxies[0].axis, a.axis)
    # Spins with direction (here along the line of sight, lightcone x): face-on disks, axis towards the observer.
    _write_lightcone(path, [[0.0, 1.0e-3, 1.0e9], [0.0, 1.0e9, 0.0]])
    with open_outputs(str(path), output_root="Lightcone") as c:
        field = lightcone_field(c, bands=(BAND,), orientation="spin")
    # Lightcone z maps to sky x, lightcone y to sky y.
    assert np.allclose(field.galaxies[0].axis, [1.0, 0.0, 0.0], atol=1.0e-6)
    assert np.allclose(field.galaxies[1].axis, [0.0, 1.0, 0.0], atol=1.0e-6)
