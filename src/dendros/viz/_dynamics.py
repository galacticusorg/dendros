"""Disk dynamics of a Galacticus galaxy, and spiral textures derived from them.

Galacticus disks are axisymmetric, so the model predicts no spiral structure. It does
predict what, in the theory of disk galaxies, shapes that structure: how strongly the
disk's self-gravity contributes to its rotation, the shear of its rotation curve, and
the surface densities of its stars and gas.  :func:`disk_dynamics` measures these at
a characteristic radius, from the model's own rotation curves (the ``rotationCurve``
node property extractor) and density grids, and :func:`dynamical_spiral_texture` maps
them onto the parameters of a :class:`~dendros.viz.SpiralTexture`:

* **Number of arms** — swing amplification favors the azimuthal wavenumber
  :math:`m = \\kappa^2 R / (2\\pi G \\Sigma_\\mathrm{disk} X)` with
  :math:`X \\approx 1.5` (Toomre 1981; Carlberg & Freedman 1985; D'Onghia, Vogelsberger
  & Hernquist 2013).  Massive, self-gravitating disks grow few arms; disks dominated by
  their halos grow many, short-lived ones, so a large :math:`m` gives a flocculent
  pattern.
* **Pitch angle** — set by the shear :math:`\\Gamma = -\\mathrm{d}\\ln\\Omega/\\mathrm{d}\\ln R`,
  :math:`\\tan p = (2/7)\\sqrt{4 - 2\\Gamma}/\\Gamma` (Michikoshi & Kokubo 2014): rising
  rotation curves give open arms, falling (bulge-dominated) ones tightly wound arms.
* **Arm strength in old stars** — proportional to the disk's share of the gravity,
  :math:`f_\\mathrm{disk} = V_\\mathrm{disk}^2/V_\\mathrm{c}^2`, so disks embedded in
  dominant halos (or bulges) are nearly smooth in old light.
* **Arm strength in young stars and dust, and clumpiness** — set by the Toomre parameter
  of the gas, :math:`Q_\\mathrm{gas} = \\sigma_\\mathrm{gas}\\kappa/(\\pi G\\Sigma_\\mathrm{gas})`:
  gas-rich, unstable disks concentrate their star formation into arms and clumps;
  gas-poor disks are smooth (lenticular-like).
* **Clump size** — a fixed fraction of the Toomre length of the gas,
  :math:`\\lambda_\\mathrm{T} = 2\\pi\\sigma_\\mathrm{gas}/\\kappa`, where the gas
  velocity dispersion is that needed to keep the gas marginally stable
  (:math:`Q_\\mathrm{gas} \\ge 1`; e.g. Krumholz & Burkert 2010), or a floor of
  10 km/s.  Gas-rich disks so get the giant clumps seen at high redshift.

Galacticus models neither velocity dispersions nor bars or tidal perturbers, so the
mapping uses the gas's floor velocity dispersion, and grand-design patterns driven by
bars or companions are not predicted.  The texture remains a presentation device; what
the model now controls is which galaxies look grand-design, multi-armed, flocculent,
clumpy or smooth, and how tightly wound their arms are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ._profiles import _trapezoid

#: Gravitational constant [Mpc (km/s)² / M☉].
GRAVITATIONAL_CONSTANT = 4.30091727e-9


@dataclass
class RotationCurve:
    """Circular velocities of a galaxy, per component, read from the ``rotationCurve`` extractor.

    ``radius`` and ``velocity`` map a component name (``"all"`` for the total, ``"disk"``,
    ``"spheroid"``, ...) to increasing radii [Mpc] and the circular velocity [km/s] there.
    """

    radius: Dict[str, np.ndarray] = field(default_factory=dict)
    velocity: Dict[str, np.ndarray] = field(default_factory=dict)

    def __contains__(self, component: str) -> bool:
        return component in self.radius and self.radius[component].size > 0

    def __call__(self, component: str, radius: float) -> float:
        """The circular velocity of ``component`` at ``radius`` [Mpc], interpolated in log-log."""
        r, v = self.radius[component], self.velocity[component]
        if r.size == 1:
            return float(v[0])
        return float(
            np.exp(np.interp(np.log(radius), np.log(r), np.log(np.maximum(v, 1.0e-30))))
        )

    def slope(self, component: str, radius: float) -> float:
        """The logarithmic slope d ln V / d ln R of ``component``'s rotation curve at ``radius``."""
        r, v = self.radius[component], self.velocity[component]
        if r.size < 2:
            return 0.0
        log_r, log_v = np.log(r), np.log(np.maximum(v, 1.0e-30))
        gradient = np.gradient(log_v, log_r, edge_order=1)
        return float(np.interp(np.log(radius), log_r, gradient))


def rotation_curves(velocities, radii, descriptions) -> RotationCurve:
    """Build the :class:`RotationCurve` of one galaxy from its row of the ``rotationCurve`` extractor.

    ``velocities`` and ``radii`` are the galaxy's rows of ``rotationCurve`` and
    ``rotationCurveRadius``; ``descriptions`` is ``rotationCurveColumns``, whose entries are
    radius specifiers (``type:component:massType:factor``).  Columns with undefined radii
    are dropped.
    """
    curves = RotationCurve()
    groups: Dict[str, list] = {}
    for k, d in enumerate(descriptions):
        text = d.decode() if isinstance(d, bytes) else str(d)
        component = text.split()[0].split(":")[1]
        if radii[k] > 0.0 and np.isfinite(radii[k]) and velocities[k] >= 0.0:
            groups.setdefault(component, []).append((radii[k], velocities[k]))
    for component, points in groups.items():
        points.sort()
        curves.radius[component] = np.array([p[0] for p in points], dtype=float)
        curves.velocity[component] = np.array([p[1] for p in points], dtype=float)
    return curves


def surface_density(table, radius: float, n_z: int = 400) -> float:
    """The surface density [M☉/Mpc²] of a :class:`~dendros.viz.DensityTable` at cylindrical ``radius`` [Mpc].

    The density is integrated through the full thickness of the component (both sides of
    its midplane).
    """
    if table is None or table.is_empty:
        return 0.0
    z = np.concatenate([[0.0], np.geomspace(1.0e-4 * radius, 20.0 * radius, n_z)])
    r = np.hypot(radius, z)
    rho = table(r, np.arctan2(radius, z))
    return float(2.0 * _trapezoid(rho, z))


@dataclass
class DiskDynamics:
    """Dynamical state of a galaxy's disk at one radius (see the module description)."""

    #: Radius at which the quantities are evaluated [Mpc].
    radius: float
    #: Total circular velocity, and that of the disk alone [km/s].
    velocity: float
    velocity_disk: float
    #: Logarithmic slope of the total rotation curve, d ln V / d ln R.
    slope: float
    #: Surface densities of the disk's stars and gas [M☉/Mpc²].
    density_surface_stellar: float
    density_surface_gas: float
    #: Velocity dispersion floor of the gas [km/s].
    dispersion_gas: float = 10.0

    @property
    def fraction_disk(self) -> float:
        """The disk's share of the gravity: V_disk² / V_c²."""
        if self.velocity <= 0.0:
            return 0.0
        return float(min((self.velocity_disk / self.velocity) ** 2, 1.0))

    @property
    def shear(self) -> float:
        """The shear rate Γ = -d ln Ω / d ln R = 1 - d ln V / d ln R (1 for a flat rotation curve)."""
        return 1.0 - self.slope

    @property
    def epicyclic_frequency(self) -> float:
        """The epicyclic frequency κ [km/s/Mpc]."""
        return float(
            np.sqrt(2.0 * max(1.0 + self.slope, 0.0)) * self.velocity / self.radius
        )

    @property
    def toomre_gas(self) -> float:
        """The Toomre parameter of the gas, at the floor velocity dispersion."""
        if self.density_surface_gas <= 0.0:
            return np.inf
        return float(
            self.dispersion_gas
            * self.epicyclic_frequency
            / (np.pi * GRAVITATIONAL_CONSTANT * self.density_surface_gas)
        )

    def arms_swing(self, swing_parameter: float = 1.5) -> float:
        """The azimuthal wavenumber most strongly swing-amplified, m = κ²R / (2πGΣ_disk X)."""
        density = self.density_surface_stellar + self.density_surface_gas
        if density <= 0.0:
            return np.inf
        return float(
            self.epicyclic_frequency**2
            * self.radius
            / (2.0 * np.pi * GRAVITATIONAL_CONSTANT * density * swing_parameter)
        )

    @property
    def pitch_shear(self) -> float:
        """The pitch angle [degrees] of swing-amplified arms for this shear (Michikoshi & Kokubo 2014)."""
        gamma = float(np.clip(self.shear, 0.05, 1.95))
        return float(
            np.rad2deg(np.arctan(2.0 / 7.0 * np.sqrt(4.0 - 2.0 * gamma) / gamma))
        )

    def dispersion_gas_effective(self, toomre_regulated: float = 1.0) -> float:
        """The gas velocity dispersion [km/s]: the floor, or that which makes Q_gas = ``toomre_regulated`` if larger."""
        kappa = self.epicyclic_frequency
        if kappa <= 0.0:
            return self.dispersion_gas
        return float(
            max(
                self.dispersion_gas,
                toomre_regulated
                * np.pi
                * GRAVITATIONAL_CONSTANT
                * self.density_surface_gas
                / kappa,
            )
        )

    def length_toomre(self, toomre_regulated: float = 1.0) -> float:
        """The Toomre length of the gas, 2πσ_gas/κ [Mpc], at the effective gas velocity dispersion."""
        kappa = self.epicyclic_frequency
        if kappa <= 0.0:
            return np.inf
        return float(
            2.0 * np.pi * self.dispersion_gas_effective(toomre_regulated) / kappa
        )


def disk_dynamics(
    scene, radius_factor: float = 2.2, dispersion_gas: float = 10.0
) -> Optional[DiskDynamics]:
    """Measure the dynamical state of ``scene``'s disk at ``radius_factor`` disk scale radii.

    The scene must carry the model's rotation curves (``scene.rotation_curve``, read when
    the model ran the ``rotationCurve`` extractor with radius specifiers ``diskRadius:all:all:*``
    and ``diskRadius:disk:all:*`` that bracket the radius).  Surface densities come from the
    disk's density grids.  Returns ``None`` for a galaxy without a disk.  The default radius,
    2.2 scale radii, is where an exponential disk's own rotation curve peaks.
    """
    curves = getattr(scene, "rotation_curve", None)
    if curves is None or "all" not in curves or "disk" not in curves:
        raise ValueError(
            "the scene has no rotation curves: run the rotationCurve extractor with radius "
            "specifiers diskRadius:all:all:* and diskRadius:disk:all:*"
        )
    if not scene.radius_disk > 0.0:
        return None
    radius = radius_factor * scene.radius_disk
    velocity = curves("all", radius)
    if not velocity > 0.0:
        return None
    stellar = next((s.density for s in scene.stars if s.name == "disk"), None)
    gaseous = next((g.density for g in scene.gas if g.name == "disk"), None)
    return DiskDynamics(
        radius=radius,
        velocity=velocity,
        velocity_disk=curves("disk", radius),
        slope=curves.slope("all", radius),
        density_surface_stellar=surface_density(stellar, radius),
        density_surface_gas=surface_density(gaseous, radius),
        dispersion_gas=dispersion_gas,
    )


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def dynamical_spiral_texture(
    scene,
    dynamics: Optional[DiskDynamics] = None,
    swing_parameter: float = 1.5,
    toomre_regulated: float = 1.0,
    arms_flocculent: float = 12.0,
    clump_fraction: float = 0.1,
    seed: Optional[int] = None,
    **overrides,
):
    """A :class:`~dendros.viz.SpiralTexture` for ``scene`` whose look follows from its disk's dynamics.

    ``dynamics`` defaults to :func:`disk_dynamics` of the scene. The mapping (see the module
    description) is:

    * ``arms`` — the swing-amplified wavenumber, rounded, and kept within 2 to 8;
    * ``flocculence`` (and ``irregularity``) — rising from 0.15 (0.25) at m = 3 to 0.95 (0.6)
      at m = ``arms_flocculent``;
    * ``segment_length`` — falling from 0.35 to 0.12 (in ln R), and ``segment_contrast``
      rising from 1.2 to 2.4, with the flocculence, and ``segment_width`` 0.8/m radians
      (within 0.05 to 0.25): many-armed patterns break into short, narrow, separate
      segments;
    * ``pitch`` — the pitch angle for the rotation curve's shear, within 8° to 45°;
    * ``amplitude_old`` — 0.6 f_disk, at most 0.5;
    * ``amplitude_young`` — 0.95 (1 - exp(-2/Q_gas)), and no less than ``amplitude_old``;
      ``amplitude_dust`` is 0.9 times that;
    * ``clumpiness`` — 1.2 / sqrt(1 + (Q_gas/1.5)²);
    * ``clump_size`` — ``clump_fraction`` times the gas's Toomre length, within 0.02 to 0.3
      scale radii.

    The constants are chosen so that a Milky-Way-like disk gets roughly the default texture.
    Any :class:`~dendros.viz.SpiralTexture` parameter may be overridden through
    ``overrides``.  Returns ``None`` for a galaxy without a disk.
    """
    from ._texture import SpiralTexture

    if dynamics is None:
        dynamics = disk_dynamics(scene)
    if dynamics is None:
        return None
    m = dynamics.arms_swing(swing_parameter)
    s = _smoothstep((m - 3.0) / (arms_flocculent - 3.0)) if np.isfinite(m) else 1.0
    q = dynamics.toomre_gas
    amplitude_old = float(np.clip(0.6 * dynamics.fraction_disk, 0.0, 0.5))
    amplitude_young = (
        max(amplitude_old, 0.95 * (1.0 - np.exp(-2.0 / q)))
        if np.isfinite(q)
        else amplitude_old
    )
    clumpiness = 1.2 / np.sqrt(1.0 + (q / 1.5) ** 2) if np.isfinite(q) else 0.0
    clump_size = float(
        np.clip(
            clump_fraction
            * dynamics.length_toomre(toomre_regulated)
            / scene.radius_disk,
            0.02,
            0.3,
        )
    )
    parameters = dict(
        radius_scale=scene.radius_disk,
        arms=int(np.clip(round(m), 2, 8)) if np.isfinite(m) else 8,
        pitch=float(np.clip(dynamics.pitch_shear, 8.0, 45.0)),
        amplitude_old=amplitude_old,
        amplitude_young=float(amplitude_young),
        amplitude_dust=float(0.9 * amplitude_young),
        clumpiness=float(clumpiness),
        clump_size=clump_size,
        flocculence=0.15 + 0.8 * s,
        irregularity=0.25 + 0.35 * s,
        segment_length=0.35 - 0.23 * s,
        segment_contrast=1.2 + 1.2 * s,
        segment_width=float(np.clip(0.8 / m, 0.05, 0.25)) if np.isfinite(m) else 0.05,
        seed=int(scene.node_index) % 100000 if seed is None else seed,
    )
    parameters.update(overrides)
    return SpiralTexture(**parameters)
