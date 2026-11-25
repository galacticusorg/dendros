"""Tests for the disk dynamics behind dynamical spiral textures in dendros.viz.

The references are analytic: the surface density of a sech^2 exponential disk, and the
epicyclic frequency, shear, swing-amplified wavenumber and pitch angle of flat and
power-law rotation curves.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("astropy")

from dendros.viz import (  # noqa: E402
    DiskDynamics,
    GasComponent,
    LightconeField,
    LightconeGalaxy,
    RotationCurve,
    SpiralTexture,
    disk_dynamics,
    dynamical_spiral_texture,
    render_lightcone,
    rotation_curves,
)
from dendros.viz._dynamics import (  # noqa: E402
    GRAVITATIONAL_CONSTANT,
    surface_density,
)

from .test_viz import MASS, RS, _disk_table, _scene  # noqa: E402

FACTORS = np.array([0.5, 0.75, 1.0, 1.5, 2.2, 3.0, 4.0])


def _curves(velocity_total, velocity_disk):
    """A rotation curve sampled at FACTORS disk scale radii from two functions of radius [Mpc]."""
    r = FACTORS * RS
    return RotationCurve(
        radius={"all": r, "disk": r},
        velocity={"all": velocity_total(r), "disk": velocity_disk(r)},
    )


def _dynamics(
    slope=0.0, velocity=200.0, fraction=0.4, sigma_star=2.0e14, sigma_gas=2.0e13
):
    radius = 2.2 * RS
    return DiskDynamics(
        radius=radius,
        velocity=velocity,
        velocity_disk=velocity * np.sqrt(fraction),
        slope=slope,
        density_surface_stellar=sigma_star,
        density_surface_gas=sigma_gas,
    )


def test_surface_density_of_exponential_disk():
    table = _disk_table()
    for factor in (0.5, 1.0, 2.2, 4.0):
        radius = factor * RS
        expected = MASS / (2.0 * np.pi * RS**2) * np.exp(-factor)
        assert surface_density(table, radius) == pytest.approx(expected, rel=0.01)


def test_rotation_curves_from_columns():
    descriptions = [
        b"diskRadius:all:all:2.2",
        b"diskRadius:all:all:1.0",
        b"diskRadius:disk:all:1.0",
        b"diskRadius:disk:all:2.2",
        b"diskRadius:spheroid:all:1.0",
    ]
    velocities = np.array([210.0, 180.0, 120.0, 140.0, -1.0])
    radii = np.array(
        [6.6e-3, 3.0e-3, 3.0e-3, 6.6e-3, -1.0]
    )  # spheroid radius undefined
    curves = rotation_curves(velocities, radii, descriptions)
    assert "spheroid" not in curves
    assert np.allclose(curves.radius["all"], [3.0e-3, 6.6e-3])
    assert np.allclose(curves.velocity["all"], [180.0, 210.0])
    assert curves("disk", 3.0e-3) == pytest.approx(120.0)
    # A power law V ∝ R^a is interpolated exactly in log-log, and its slope is a.
    a = np.log(210.0 / 180.0) / np.log(2.2)
    assert curves("all", 4.5e-3) == pytest.approx(180.0 * 1.5**a)
    assert curves.slope("all", 4.5e-3) == pytest.approx(a)


def test_flat_rotation_curve():
    """Γ = 1, κ = √2 V/R, and the Michikoshi & Kokubo pitch angle tan p = (2/7)√2."""
    d = _dynamics(slope=0.0)
    assert d.shear == pytest.approx(1.0)
    assert d.epicyclic_frequency == pytest.approx(np.sqrt(2.0) * d.velocity / d.radius)
    assert d.pitch_shear == pytest.approx(
        np.rad2deg(np.arctan(2.0 / 7.0 * np.sqrt(2.0)))
    )
    kappa = d.epicyclic_frequency
    m = kappa**2 * d.radius / (2.0 * np.pi * GRAVITATIONAL_CONSTANT * 2.2e14 * 1.5)
    assert d.arms_swing() == pytest.approx(m)
    q = 10.0 * kappa / (np.pi * GRAVITATIONAL_CONSTANT * 2.0e13)
    assert d.toomre_gas == pytest.approx(q)
    assert d.fraction_disk == pytest.approx(0.4)


def test_pitch_follows_shear():
    """Rising rotation curves (low shear) wind arms more openly than falling ones."""
    pitches = [_dynamics(slope=s).pitch_shear for s in (0.5, 0.0, -0.3)]
    assert pitches[0] > pitches[1] > pitches[2]


def test_gas_dispersion_regulated_to_marginal_stability():
    # Gas so dense that Q < 1 at the floor dispersion: the dispersion rises until Q = 1.
    d = _dynamics(sigma_gas=5.0e14)
    assert d.toomre_gas < 1.0
    sigma = d.dispersion_gas_effective(1.0)
    assert sigma > d.dispersion_gas
    assert sigma * d.epicyclic_frequency / (
        np.pi * GRAVITATIONAL_CONSTANT * d.density_surface_gas
    ) == pytest.approx(1.0)
    assert d.length_toomre() == pytest.approx(
        2.0 * np.pi * sigma / d.epicyclic_frequency
    )
    # Stable gas keeps the floor.
    assert _dynamics(sigma_gas=1.0e12).dispersion_gas_effective() == pytest.approx(10.0)


def test_disk_dynamics_of_scene():
    """disk_dynamics reads the scene's rotation curves and density grids."""
    scene = _scene(
        gas=[GasComponent("disk", _disk_table(MASS / 5.0, "gaseous"), MASS / 5.0, 0.01)]
    )
    scene.rotation_curve = _curves(
        lambda r: np.full(r.shape, 220.0), lambda r: 150.0 * (r / (2.2 * RS)) ** 0.1
    )
    d = disk_dynamics(scene)
    assert d.radius == pytest.approx(2.2 * RS)
    assert d.velocity == pytest.approx(220.0)
    assert d.velocity_disk == pytest.approx(150.0)
    assert d.slope == pytest.approx(0.0, abs=1.0e-12)
    expected = MASS / (2.0 * np.pi * RS**2) * np.exp(-2.2)
    assert d.density_surface_stellar == pytest.approx(expected, rel=0.01)
    assert d.density_surface_gas == pytest.approx(expected / 5.0, rel=0.01)


def test_disk_dynamics_needs_rotation_curves():
    with pytest.raises(ValueError, match="rotationCurve"):
        disk_dynamics(_scene())


def test_texture_follows_dynamics():
    scene = _scene()
    # A massive, self-gravitating disk: few, strong arms; a halo-dominated one: many, flocculent, weak in old light.
    heavy = dynamical_spiral_texture(scene, _dynamics(fraction=0.7, sigma_star=6.0e14))
    light = dynamical_spiral_texture(scene, _dynamics(fraction=0.1, sigma_star=3.0e13))
    assert isinstance(heavy, SpiralTexture)
    assert heavy.arms < light.arms
    assert heavy.flocculence < light.flocculence
    assert heavy.amplitude_old > light.amplitude_old
    assert heavy.segment_length > light.segment_length
    assert heavy.segment_contrast < light.segment_contrast
    # The arms fade beyond the radius at which the dynamics were evaluated.
    assert heavy.radius_outer == pytest.approx(2.2)
    # Gas-rich (unstable) disks are clumpier, with larger clumps, than gas-poor ones; a gas-free disk has no clumps
    # and no extra concentration of young light into the arms.
    rich = dynamical_spiral_texture(scene, _dynamics(sigma_gas=5.0e14))
    poor = dynamical_spiral_texture(scene, _dynamics(sigma_gas=2.0e12))
    none = dynamical_spiral_texture(scene, _dynamics(sigma_gas=0.0))
    assert rich.clumpiness > poor.clumpiness > none.clumpiness == 0.0
    assert rich.clump_size > poor.clump_size
    assert rich.amplitude_young > poor.amplitude_young
    assert none.amplitude_young == pytest.approx(none.amplitude_old)
    # Overrides win.
    assert dynamical_spiral_texture(scene, _dynamics(), arms=3, seed=5).arms == 3


def test_dynamical_lightcone_texture_needs_rotation_curves():
    field = LightconeField(
        [LightconeGalaxy(_scene(), 1.0, 1.0, np.zeros(2), np.array([0.0, 0.0, 1.0]))],
        20.0,
        dict(
            HubbleConstant=73.0,
            OmegaMatter=0.25,
            OmegaDarkEnergy=0.75,
            OmegaBaryon=0.045,
        ),
    )
    with pytest.raises(ValueError, match="rotation curves"):
        render_lightcone(field, texture="dynamical")
    with pytest.raises(ValueError, match="texture must be"):
        render_lightcone(field, texture="spiral")
