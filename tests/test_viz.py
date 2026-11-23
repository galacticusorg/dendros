"""Tests for dendros.viz: density tables, the ray-marched renderer, and the texture.

The references are analytic: a sech^2 exponential disk, whose total light, face-on
central surface brightness, and attenuation by uniformly mixed dust are known in
closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("astropy")

from dendros.viz._profiles import _trapezoid  # noqa: E402

from dendros.viz import (  # noqa: E402
    DensityTable,
    DustModel,
    GalaxyScene,
    GasComponent,
    SpiralTexture,
    StellarComponent,
    compose_rgb,
    density_tables,
    render_galaxy,
    rotation_matrix,
)

RS = 3.0e-3  # disk scale radius [Mpc]
ZS = 0.137 * RS  # sech^2 scale height [Mpc]
MASS = 5.0e10
LUMINOSITY = 2.0e10
# The grid used by the prototype model: 40 radii, 22 polar angles packed towards the midplane.
FACTORS = np.geomspace(0.01, 10.0, 40)
OFFSETS = np.array(
    [
        0.0,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        11.0,
        15.0,
        20.0,
        27.0,
        35.0,
        45.0,
        55.0,
        65.0,
        75.0,
        90.0,
    ]
)


def _disk_density(r, theta, mass=MASS):
    R = r * np.sin(theta)
    z = r * np.cos(theta)
    return mass / (4.0 * np.pi * RS**2 * ZS) * np.exp(-R / RS) / np.cosh(z / ZS) ** 2


def _disk_table(mass=MASS, mass_type="stellar"):
    r = FACTORS * RS
    theta = np.deg2rad(np.sort(90.0 - OFFSETS))
    return DensityTable(
        "disk", mass_type, r, theta, _disk_density(r[:, None], theta[None, :], mass)
    )


def _scene(gas=None):
    disk = StellarComponent("disk", _disk_table(), MASS, {"SDSS_r": LUMINOSITY})
    return GalaxyScene([disk], gas or [], ["SDSS_r"], 1, 1, RS, 0.0, 1.68 * RS)


def test_table_interpolation_matches_analytic_disk():
    table = _disk_table()
    rng = np.random.default_rng(3)
    r = RS * 10.0 ** rng.uniform(-1.0, 0.7, 2000)
    theta = np.deg2rad(rng.uniform(60.0, 120.0, 2000))
    rho = table(r, theta)
    exact = _disk_density(r, theta)
    significant = exact > 1.0e-3 * exact.max()
    assert np.max(np.abs(rho[significant] / exact[significant] - 1.0)) < 0.05
    # Beyond the table the density is zero.
    assert table(np.array([20.0 * RS]), np.array([np.pi / 2]))[0] == 0.0


def test_table_mass():
    expected = 1.0 - 11.0 * np.exp(
        -10.0
    )  # fraction of an exponential disk inside R = 10 r_s
    assert _disk_table().mass() / MASS == pytest.approx(expected, rel=2.0e-3)


@pytest.mark.parametrize("inclination", [0.0, 60.0, 85.0])
def test_light_is_conserved(inclination):
    half_width = 11.0 * RS
    image = render_galaxy(
        _scene(),
        inclination=inclination,
        npix=128,
        half_width=half_width,
        n_steps=512,
        dust=DustModel(enabled=False),
    )["SDSS_r"]
    pixel_area = (2.0 * half_width / 128 * 1.0e6) ** 2  # pc^2
    assert image.sum() * pixel_area / LUMINOSITY == pytest.approx(
        1.0 - 11.0 * np.exp(-10.0), rel=3.0e-3
    )


def test_face_on_central_surface_brightness():
    image = render_galaxy(
        _scene(),
        inclination=0.0,
        npix=65,
        half_width=6.0 * RS,
        n_steps=1024,
        dust=DustModel(enabled=False),
    )["SDSS_r"]
    expected = LUMINOSITY / (2.0 * np.pi * (RS * 1.0e6) ** 2)
    assert image[32, 32] / expected == pytest.approx(1.0, rel=2.0e-3)


def test_mixed_dust_attenuation():
    """Stars and dust with the same profile: the face-on center is attenuated by (1-e^-tau)/tau."""
    metallicity, mass_gas = 0.02, 1.0e10
    dust = DustModel(dust_to_metals=0.4)
    gas = GasComponent("disk", _disk_table(mass_gas, "gaseous"), mass_gas, metallicity)
    kwargs = dict(inclination=0.0, npix=65, half_width=6.0 * RS, n_steps=1024)
    clear = render_galaxy(_scene(), dust=DustModel(enabled=False), **kwargs)["SDSS_r"]
    dusty = render_galaxy(_scene([gas]), dust=dust, **kwargs)["SDSS_r"]
    tau = (
        dust.kappa("SDSS_r")
        * dust.dust_to_metals
        * metallicity
        * mass_gas
        / (2.0 * np.pi * RS**2)
    )
    assert dusty[32, 32] / clear[32, 32] == pytest.approx(
        -np.expm1(-tau) / tau, rel=2.0e-3
    )


def _young_scene(fraction=0.5):
    disk = StellarComponent(
        "disk",
        _disk_table(),
        MASS,
        {"SDSS_r": LUMINOSITY},
        {"SDSS_r": fraction * LUMINOSITY},
    )
    return GalaxyScene([disk], [], ["SDSS_r"], 1, 1, RS, 0.0, 1.68 * RS)


@pytest.mark.parametrize("inclination", [0.0, 70.0])
def test_thin_young_layer_conserves_light(inclination):
    """Squeezing the young stars into a thinner layer moves light vertically but conserves it."""
    half_width = 11.0 * RS
    kwargs = dict(
        inclination=inclination,
        npix=128,
        half_width=half_width,
        n_steps=2048,
        dust=DustModel(enabled=False),
    )
    thick = render_galaxy(_young_scene(), **kwargs)["SDSS_r"]
    thin = render_galaxy(_young_scene(), young_scale_height={"disk": 0.3}, **kwargs)[
        "SDSS_r"
    ]
    assert thin.sum() / thick.sum() == pytest.approx(1.0, rel=1.0e-3)
    if inclination == 0.0:
        # Face-on, each line of sight crosses the whole layer: the image is unchanged.
        assert np.allclose(thin[48:80, 48:80], thick[48:80, 48:80], rtol=2.0e-3)
    else:
        # Inclined, the thinner layer concentrates light towards the disk's major axis.
        assert thin[64, 64] > thick[64, 64]


def test_texture_arms_are_irregular():
    """The arms' phase wanders: the azimuth of peak arm brightness does not follow a perfect logarithmic spiral."""
    phi = np.linspace(-np.pi, np.pi, 4096, endpoint=False)

    def offsets(texture):
        result = []
        for R in RS * np.geomspace(1.0, 4.0, 12):
            factor = texture(R * np.cos(phi), R * np.sin(phi), 0.0 * phi, "disk", "old")
            ideal = (
                -2.0 * np.log(R / RS) / np.tan(np.deg2rad(texture.pitch))
            )  # arms * phi - u/tan(pitch) = const
            spectrum = np.sum(factor * np.exp(-2j * phi))
            result.append(np.angle(spectrum * np.exp(-1j * ideal)))
        return np.unwrap(np.array(result))

    regular = offsets(
        SpiralTexture(
            radius_scale=RS, clumpiness=0.0, irregularity=0.0, flocculence=0.0
        )
    )
    irregular = offsets(
        SpiralTexture(
            radius_scale=RS, clumpiness=0.0, irregularity=0.6, flocculence=0.0
        )
    )
    assert np.std(regular) < 1.0e-3
    assert np.std(irregular) > 0.05


def test_rotation_matrix():
    # Face-on: the disk axis points along the line of sight.
    assert np.allclose(rotation_matrix(0.0) @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    # Edge-on: the line of sight lies in the disk plane, and the disk lies along the image x axis.
    m = rotation_matrix(90.0)
    assert abs((m @ [0.0, 0.0, 1.0])[2]) < 1.0e-12
    assert np.allclose(m @ [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert np.allclose(m @ m.T, np.eye(3))


def test_texture_preserves_annulus_light():
    texture = SpiralTexture(radius_scale=RS, clumpiness=0.0)
    # Sample 16 points per cell of the texture's azimuthal grid: the mean of the linear interpolant is then exact.
    phi = -np.pi + 2.0 * np.pi * np.arange(16 * texture._N_PHI) / (16 * texture._N_PHI)
    for R in (0.3 * RS, RS, 3.0 * RS):
        x, y = R * np.cos(phi), R * np.sin(phi)
        for kind in ("young", "old", "dust"):
            assert texture(x, y, 0.0 * x, "disk", kind).mean() == pytest.approx(
                1.0, rel=1.0e-6
            )
    assert np.all(
        texture(np.ones(3), np.ones(3), np.zeros(3), "spheroid", "young") == 1.0
    )


def test_density_tables_from_columns():
    """Columns ordered with radius fastest, as the densityProfile extractor writes them."""
    specs = [f"diskRadius:disk:stellar:{f:.5g}" for f in FACTORS]
    angles = np.sort(90.0 - OFFSETS)[::-1]  # any order
    descriptions, values, radii, rho = [], [], [], []
    for a in angles:
        for f, sp in zip(FACTORS, specs):
            descriptions.append(f"{sp} θ={a:9.4f}°".encode())
            values.append(a)
            radii.append(f * RS)
            rho.append(_disk_density(f * RS, np.deg2rad(a)))
    tables = density_tables(
        np.array(rho), np.array(radii), np.array(descriptions), np.array(values)
    )
    table = tables[("disk", "stellar")]
    assert table.density.shape == (FACTORS.size, OFFSETS.size)
    assert np.all(np.diff(table.angle) > 0.0)
    assert table.mass() / MASS == pytest.approx(1.0 - 11.0 * np.exp(-10.0), rel=2.0e-3)


def test_compose_rgb_shape():
    images = {b: np.full((8, 8), 1.0) for b in ("SDSS_g", "SDSS_r", "SDSS_i")}
    rgb = compose_rgb(images)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8


def test_thin_dust_layer():
    """A dust layer thinner than the stars (at the same column density), seen face-on, against an exact quadrature
    along the central line of sight."""
    metallicity, mass_gas, f = 0.02, 1.0e10, 0.2
    dust = DustModel(dust_to_metals=0.4, scale_height_factor=f)
    gas = GasComponent("disk", _disk_table(mass_gas, "gaseous"), mass_gas, metallicity)
    kwargs = dict(inclination=0.0, npix=65, half_width=6.0 * RS, n_steps=4096)
    clear = render_galaxy(_scene(), dust=DustModel(enabled=False), **kwargs)["SDSS_r"]
    dusty = render_galaxy(_scene([gas]), dust=dust, **kwargs)["SDSS_r"]
    # Exact: I = int rho_*(z) exp(-tau(z)) dz, tau(z) = kappa int_z^inf rho_dust dz', along the central line of sight.
    z = np.linspace(6.0 * RS, -6.0 * RS, 200001)
    rho_stars = 1.0 / np.cosh(z / ZS) ** 2
    rho_dust = 1.0 / np.cosh(z / (f * ZS)) ** 2
    tau_total = (
        dust.kappa("SDSS_r")
        * dust.dust_to_metals
        * metallicity
        * mass_gas
        / (2.0 * np.pi * RS**2)
    )
    tau = (
        tau_total
        * np.concatenate([[0.0], np.cumsum(0.5 * (rho_dust[1:] + rho_dust[:-1]))])
        / _trapezoid(rho_dust)
    )
    expected = _trapezoid(rho_stars * np.exp(-tau)) / _trapezoid(rho_stars)
    assert expected < 0.75  # the layer really does attenuate
    assert dusty[32, 32] / clear[32, 32] == pytest.approx(expected, rel=3.0e-3)


def test_clump_field_has_unit_mean():
    texture = SpiralTexture(radius_scale=RS, amplitude_young=0.0)
    xy = np.linspace(-5.0 * RS, 5.0 * RS, 801)
    X, Y = np.meshgrid(xy, xy)
    factor = texture(X, Y, 0.0 * X, "disk", "young")
    assert factor.std() > 0.3  # the clumps are really there
    assert factor.mean() == pytest.approx(1.0, abs=0.05)


def test_white_balance_makes_reference_color_neutral():
    g_r, r_i = 0.6, 0.3
    images = {"SDSS_r": np.full((4, 4), 1.0)}
    images["SDSS_g"] = images["SDSS_r"] * 10.0 ** (-0.4 * g_r)
    images["SDSS_i"] = images["SDSS_r"] * 10.0 ** (0.4 * r_i)
    rgb = compose_rgb(images, white=(g_r, r_i)).astype(int)
    assert np.all(np.abs(rgb[..., 0] - rgb[..., 2]) <= 1) and np.all(
        np.abs(rgb[..., 0] - rgb[..., 1]) <= 1
    )
