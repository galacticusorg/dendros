"""Optional, *non-model* textures for galaxy images.

Galacticus disks are axisymmetric: the model predicts their radial and vertical
profiles, masses, luminosities and colors, but not spiral structure.  The texture
here adds spiral arms and clumpy star formation for presentation images.  It
redistributes light and dust within the disk plane. The arm factor averages to exactly
one around every annulus; the clump field (a log-normal random field with unit mean)
averages to one only statistically, so the azimuthally averaged profiles, and the
total luminosities and masses, stay close to the model's ones.

The arms are a mixture of a grand-design pattern - logarithmic spirals whose phase
wanders and whose brightness varies along their length - and a flocculent one, made of
short spiral segments: a random field stretched along the spiral direction.

Its look is driven by quantities the model does provide: light from recently formed
stars is concentrated into the arms and knots much more strongly than older light,
so arm colors follow from the model's own young-to-old luminosity ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SpiralTexture:
    """Logarithmic spiral arms with clumpy star formation.

    Parameters
    ----------
    radius_scale:
        Disk scale radius [Mpc] (``GalaxyScene.radius_disk``); the arm geometry is set in
        these units.
    arms:
        Number of arms.
    pitch:
        Pitch angle of the arms [degrees].
    sharpness:
        Arm narrowness: the arm profile in phase is :math:`[(1+\\cos\\psi)/2]^k`.
    amplitude_young, amplitude_old, amplitude_dust:
        Fraction (0 to 1) of the light (or dust) at each radius that is gathered into the
        arms. The rest stays smooth.
    clumpiness:
        Standard deviation of the log-normal clump field applied to young light (and,
        at half strength, to dust). Zero for smooth arms.
    clump_size:
        Correlation length of the clump field [scale radii].
    radius_inner:
        Radius [scale radii] inside which the arms fade out (bulge region).
    radius_outer:
        Radius [scale radii] beyond which the arms fade out, as a Gaussian in
        (R - radius_outer) of width ``radius_outer_width`` scale radii. ``None`` (the
        default) keeps the arms at full amplitude out to any radius.
    radius_outer_width:
        Width [scale radii] of the outer fade.
    components:
        Stellar/gas components to which the texture applies.
    irregularity:
        Irregularity of the grand-design arms: the rms wander of their phase [radians],
        and the rms (natural log) variation of their brightness along their length.
        Zero for perfect logarithmic spirals.
    flocculence:
        Fraction (0 to 1) of the arm pattern made of short spiral segments rather than
        grand-design arms.
    segment_length:
        Correlation length of the flocculent segments, and of the variations along
        the grand-design arms, in ln(R).
    segment_width:
        Correlation length of the flocculent segments across the spiral direction
        [radians of azimuth].
    segment_contrast:
        Standard deviation of the log-normal field of flocculent segments. Larger values
        make the segments sparser, separated by dark gaps rather than joined into arcs.
    seed:
        Random seed for the clump field and the arm pattern.
    """

    radius_scale: float
    arms: int = 2
    pitch: float = 14.0
    sharpness: float = 6.0
    amplitude_young: float = 0.9
    amplitude_old: float = 0.3
    amplitude_dust: float = 0.8
    clumpiness: float = 0.8
    clump_size: float = 0.06
    radius_inner: float = 0.5
    radius_outer: Optional[float] = None
    radius_outer_width: float = 1.0
    components: Sequence[str] = ("disk",)
    irregularity: float = 0.35
    flocculence: float = 0.35
    segment_length: float = 0.35
    segment_width: float = 0.12
    segment_contrast: float = 1.2
    seed: int = 1

    #: Range of ln(R/radius_scale) covered by the arm tables, and their resolution.
    _U_RANGE = (-6.0, 4.0)
    _N_U = 1024
    _N_PHI = 1024
    #: Half-width [scale radii] and resolution of the (Cartesian) clump field.
    _CLUMP_EXTENT = 12.0
    _N_CLUMP = 2048

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._phase = rng.uniform(0.0, 2.0 * np.pi)
        self._tables = {}
        self._pattern = self._arm_pattern(rng)
        # A Gaussian random field in the disk plane, with unit variance, whose power is concentrated on scales
        # between clump_size and ten times that. Its size is set in physical units, so the clumps are the same size
        # at all radii.
        n = self._N_CLUMP
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=2.0 * self._CLUMP_EXTENT / n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        kk = np.hypot(kx, ky)
        k_clump = 1.0 / self.clump_size  # correlation length = clump_size
        amplitude = np.exp(-0.5 * (kk / k_clump) ** 2) * (
            1.0 - np.exp(-0.5 * (10.0 * kk / k_clump) ** 2)
        )
        noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        field = np.fft.ifft2(noise * amplitude).real
        self._clumps = (field - field.mean()) / field.std()

    def _random_field(self, rng, length_u, length_eta):
        """A Gaussian random field with unit variance on the (u, eta) grid of the arm tables, with Gaussian
        correlations of length ``length_u`` in u = ln(R/radius_scale) and ``length_eta`` in eta (periodic, 2 pi).
        """
        k_u = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                self._N_U, d=(self._U_RANGE[1] - self._U_RANGE[0]) / self._N_U
            )
        )
        k_eta = np.fft.fftfreq(
            self._N_PHI, d=1.0 / self._N_PHI
        )  # integer wavenumbers: periodic in 2 pi
        KU, KE = np.meshgrid(k_u, k_eta, indexing="ij")
        amplitude = np.exp(-0.25 * ((KU * length_u) ** 2 + (KE * length_eta) ** 2))
        noise = rng.standard_normal(
            (self._N_U, self._N_PHI)
        ) + 1j * rng.standard_normal((self._N_U, self._N_PHI))
        field = np.fft.ifft2(noise * amplitude).real
        return (field - field.mean()) / field.std()

    def _arm_pattern(self, rng):
        """The arm pattern on the (u, phi) grid, normalized to unit mean around every annulus.

        Fields are built in (u, eta), where eta = phi - u/tan(pitch) is constant along a spiral, so that structures
        elongated in u follow the spiral direction.
        """
        u = np.linspace(*self._U_RANGE, self._N_U)
        eta = np.linspace(-np.pi, np.pi, self._N_PHI, endpoint=False)
        U, ETA = np.meshgrid(u, eta, indexing="ij")
        # Grand-design arms: the phase wanders slowly, and the brightness varies along each arm.
        wander = self._random_field(rng, 2.0 * self.segment_length, 0.8)
        variation = self._random_field(rng, self.segment_length, 0.4)
        psi = self.arms * ETA + self._phase + self.irregularity * wander
        grand = (0.5 * (1.0 + np.cos(psi))) ** self.sharpness * np.exp(
            self.irregularity * variation
        )
        grand /= grand.mean(axis=1, keepdims=True)
        # Flocculent segments: a log-normal field elongated along the spiral direction.
        flocculent = np.exp(
            self.segment_contrast
            * self._random_field(rng, self.segment_length, self.segment_width)
        )
        flocculent /= flocculent.mean(axis=1, keepdims=True)
        pattern_eta = (1.0 - self.flocculence) * grand + self.flocculence * flocculent
        # Resample from eta onto phi = eta + u/tan(pitch), row by row (a periodic shift of each row).
        shift = u / np.tan(np.deg2rad(self.pitch))
        phi = np.linspace(-np.pi, np.pi, self._N_PHI + 1)  # endpoint included: periodic
        position = (
            (phi[None, :] - shift[:, None] + np.pi) / (2.0 * np.pi) * self._N_PHI
        ) % self._N_PHI
        j0 = position.astype(np.intp) % self._N_PHI
        j1 = (j0 + 1) % self._N_PHI
        w = position - np.floor(position)
        rows = np.arange(self._N_U)[:, None]
        pattern = (1.0 - w) * pattern_eta[rows, j0] + w * pattern_eta[rows, j1]
        # Linear interpolation of a periodic row onto a shifted copy of its grid keeps the mean of the row only
        # approximately: renormalize so that every annulus averages to exactly one.
        pattern /= pattern[:, :-1].mean(axis=1, keepdims=True)
        return pattern

    def _arm_table(self, kind):
        """The arm factor for ``kind`` on a grid in (ln R/radius_scale, phi), built on first use."""
        if kind not in self._tables:
            amplitude = {
                "young": self.amplitude_young,
                "dust": self.amplitude_dust,
            }.get(kind, self.amplitude_old)
            R = self.radius_scale * np.exp(np.linspace(*self._U_RANGE, self._N_U))
            # Fade the arms in outside radius_inner so the bulge region stays smooth.
            fade = 1.0 - np.exp(-((R / (self.radius_inner * self.radius_scale)) ** 2))
            # Fade them out beyond radius_outer, if set.
            if self.radius_outer is not None:
                excess = np.maximum(R / self.radius_scale - self.radius_outer, 0.0)
                fade = fade * np.exp(-0.5 * (excess / self.radius_outer_width) ** 2)
            a = (amplitude * fade)[:, None]
            self._tables[kind] = (1.0 - a) + a * self._pattern
        return self._tables[kind]

    @staticmethod
    def _bilinear(table, fi, fj):
        i0 = np.clip(fi.astype(np.intp), 0, table.shape[0] - 2)
        j0 = np.clip(fj.astype(np.intp), 0, table.shape[1] - 2)
        a = np.clip(fi - i0, 0.0, 1.0)
        b = np.clip(fj - j0, 0.0, 1.0)
        return (1.0 - a) * ((1.0 - b) * table[i0, j0] + b * table[i0, j0 + 1]) + a * (
            (1.0 - b) * table[i0 + 1, j0] + b * table[i0 + 1, j0 + 1]
        )

    def _clump_factor(self, x, y, sigma):
        if sigma <= 0.0:
            return 1.0
        scale = (self._N_CLUMP - 1) / (2.0 * self._CLUMP_EXTENT * self.radius_scale)
        g = self._bilinear(
            self._clumps,
            (x + self._CLUMP_EXTENT * self.radius_scale) * scale,
            (y + self._CLUMP_EXTENT * self.radius_scale) * scale,
        )
        # Log-normal with unit mean (bilinear interpolation slightly reduces the variance; the mean stays close to one).
        return np.exp(sigma * g - 0.5 * sigma**2)

    def __call__(self, x, y, z, component, kind):
        if component not in self.components:
            return np.ones(np.shape(x))
        R = np.hypot(x, y)
        u = np.log(np.maximum(R, 1.0e-300) / self.radius_scale)
        fi = (
            (np.clip(u, *self._U_RANGE) - self._U_RANGE[0])
            / (self._U_RANGE[1] - self._U_RANGE[0])
            * (self._N_U - 1)
        )
        fj = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi) * self._N_PHI
        factor = self._bilinear(self._arm_table(kind), fi, fj)
        if kind == "young":
            factor = factor * self._clump_factor(x, y, self.clumpiness)
        elif kind == "dust":
            factor = factor * self._clump_factor(x, y, 0.5 * self.clumpiness)
        return factor
