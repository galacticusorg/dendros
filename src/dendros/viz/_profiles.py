"""Density tables read from the Galacticus ``densityProfile`` node property extractor.

The extractor evaluates the density of a component at a list of radius specifiers
(``type:component:massType:factor``) and, optionally, at a list of polar angles
measured from the component's symmetry axis.  Each (radius, angle) pair is one
column of the output, with the radius varying fastest.  This module turns those
columns into a :class:`DensityTable` per (component, mass type), which can then be
evaluated at arbitrary positions in the galaxy's own frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

#: Trapezoidal integration (``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0, and dendros also supports NumPy 1.x).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: Galacticus writes this sentinel where a radius (or the density at it) is undefined.
UNDEFINED = -1.0

#: Floor applied before taking logarithms, in M☉/Mpc³.  Far below any density of interest.
_DENSITY_FLOOR = 1.0e-30


@dataclass
class DensityTable:
    """The density of one component of one galaxy, tabulated on a (radius, polar angle) grid.

    Attributes
    ----------
    component, mass_type:
        As named in the radius specifiers (e.g. ``"disk"``, ``"stellar"``).
    radius:
        Radii of the grid [Mpc], increasing, shape ``(n_r,)``.
    angle:
        Polar angles of the grid [radians], increasing, in ``[0, pi/2]``, shape ``(n_theta,)``.
    density:
        Density [M☉/Mpc³], shape ``(n_r, n_theta)``.
    """

    component: str
    mass_type: str
    radius: np.ndarray
    angle: np.ndarray
    density: np.ndarray

    #: Oversampling of the lookup table relative to the tabulated grid, in each direction.
    oversample: int = 8

    def __post_init__(self) -> None:
        self._log_r = np.log(self.radius)
        self._log_rho = np.log(np.maximum(self.density, _DENSITY_FLOOR))
        self._mu = np.cos(self.angle)[::-1]  # increasing
        self._lookup = None
        if self.is_empty or self._log_r.size < 2 or self._mu.size < 2:
            return
        # Interpolate log density with a bicubic spline in (log r, cos theta).  The vertical sech^2 and exponential
        # profiles of a disk are close to linear in cos theta at fixed r, while the curvature of log density in log r
        # (e.g. -R/r_s for an exponential disk) is captured by the cubic.  Bilinear interpolation on the 40 x 22 grid
        # of the prototype model loses ~1.3% of the light of an exponential disk.  The spline is slow to evaluate, so
        # it is sampled once onto a grid `oversample` times finer than the tabulated one - uniform in the *index* of
        # the tabulated nodes, so that it stays densest where the model's grid is - and that grid is then
        # interpolated bilinearly.
        from scipy.interpolate import RectBivariateSpline

        kx = min(3, self._log_r.size - 1)
        ky = min(3, self._mu.size - 1)
        spline = RectBivariateSpline(
            self._log_r, self._mu, self._log_rho[:, ::-1], kx=kx, ky=ky, s=0
        )
        n_r = (self._log_r.size - 1) * self.oversample + 1
        n_mu = (self._mu.size - 1) * self.oversample + 1
        index_r = np.linspace(0.0, self._log_r.size - 1, n_r)
        index_mu = np.linspace(0.0, self._mu.size - 1, n_mu)
        fine_log_r = np.interp(index_r, np.arange(self._log_r.size), self._log_r)
        fine_mu = np.interp(index_mu, np.arange(self._mu.size), self._mu)
        lookup = spline(fine_log_r, fine_mu)
        self._lookup = np.minimum(lookup, self._log_rho.max())

    @property
    def is_empty(self) -> bool:
        return not np.any(self.density > 0.0)

    def __call__(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Evaluate the density at spherical radius ``r`` [Mpc] and polar angle ``theta``.

        ``theta`` is folded into ``[0, pi/2]`` (the components are symmetric about
        their equatorial plane).  Inside the innermost radius the density is held at
        its value there; beyond the outermost radius it is zero.
        """
        r = np.asarray(r, dtype=float)
        if self.is_empty:
            return np.zeros(r.shape)
        if self._lookup is None:
            # A single angle (or radius): the component is tabulated in the equatorial plane only, so treat it as
            # spherical.
            return self._spherical(r)
        theta = np.asarray(theta, dtype=float)
        rho = np.zeros(np.broadcast(r, theta).shape)
        inside = r <= self.radius[-1]
        if not np.any(inside):
            return rho
        r_in = np.broadcast_to(r, rho.shape)[inside]
        mu = np.abs(np.cos(np.broadcast_to(theta, rho.shape)[inside]))
        # Fractional positions in the lookup table.
        i = (
            np.interp(
                np.log(np.maximum(r_in, self.radius[0])),
                self._log_r,
                np.arange(self._log_r.size),
            )
            * self.oversample
        )
        j = np.interp(mu, self._mu, np.arange(self._mu.size)) * self.oversample
        n_i, n_j = self._lookup.shape
        i0 = np.minimum(i.astype(np.intp), n_i - 2)
        j0 = np.minimum(j.astype(np.intp), n_j - 2)
        fi = i - i0
        fj = j - j0
        t = self._lookup
        value = (1.0 - fi) * ((1.0 - fj) * t[i0, j0] + fj * t[i0, j0 + 1]) + fi * (
            (1.0 - fj) * t[i0 + 1, j0] + fj * t[i0 + 1, j0 + 1]
        )
        rho_in = np.exp(value)
        rho_in[rho_in <= _DENSITY_FLOOR * 1.0001] = 0.0
        rho[inside] = rho_in
        return rho

    def _spherical(self, r: np.ndarray) -> np.ndarray:
        log_rho = np.interp(
            np.log(np.maximum(r, self.radius[0])), self._log_r, self._log_rho[:, -1]
        )
        rho = np.exp(log_rho)
        rho[r > self.radius[-1]] = 0.0
        rho[rho <= _DENSITY_FLOOR * 1.0001] = 0.0
        return rho

    def mass(self, n_r: int = 400, n_mu: int = 200) -> float:
        """The mass in the table [M☉], integrated out to its outermost radius.

        Compare with the component's mass to check that the grid covers the component.
        """
        if self.is_empty:
            return 0.0
        r = np.geomspace(self.radius[0], self.radius[-1], n_r)
        mu = np.linspace(0.0, 1.0, n_mu)
        R, M = np.meshgrid(r, mu, indexing="ij")
        rho = self(R.ravel(), np.arccos(M.ravel())).reshape(R.shape)
        inner = (
            rho[0, :].mean() * 4.0 / 3.0 * np.pi * r[0] ** 3
        )  # constant-density core inside the grid
        # Both hemispheres: 4 pi \int r^2 dr \int_0^1 dmu.
        return float(
            inner + 4.0 * np.pi * _trapezoid(_trapezoid(rho, mu, axis=1) * r**2, r)
        )


def _parse_column(description: str) -> Tuple[str, str, str]:
    """Return (radius specifier, component, mass type) from a column description."""
    specifier = description.split()[0]
    fields = specifier.split(":")
    if len(fields) < 4:
        raise ValueError(
            f"cannot parse radius specifier from column description {description!r}"
        )
    return specifier, fields[1], fields[2]


def density_tables(
    densities: np.ndarray,
    radii: np.ndarray,
    descriptions,
    angles_degrees: np.ndarray,
) -> Dict[Tuple[str, str], DensityTable]:
    """Build the :class:`DensityTable` objects for one galaxy.

    Parameters
    ----------
    densities, radii:
        The galaxy's row of ``densityProfile`` and ``densityProfileRadius``, shape ``(n_columns,)``.
    descriptions:
        The ``densityProfileColumns`` dataset (bytes or str), shape ``(n_columns,)``.
    angles_degrees:
        The ``densityProfileColumnValues`` dataset: the polar angle of each column [degrees].

    Columns are grouped by (component, mass type).  Within a group the radius grid is
    the set of distinct specifiers in order of appearance, and the angle grid the set
    of distinct angles.
    """
    descriptions = [
        d.decode() if isinstance(d, bytes) else str(d) for d in descriptions
    ]
    angles_degrees = np.asarray(angles_degrees, dtype=float)
    groups: Dict[Tuple[str, str], Dict[str, list]] = {}
    for k, desc in enumerate(descriptions):
        specifier, component, mass_type = _parse_column(desc)
        g = groups.setdefault((component, mass_type), {"specifiers": [], "columns": []})
        if specifier not in g["specifiers"]:
            g["specifiers"].append(specifier)
        g["columns"].append((g["specifiers"].index(specifier), angles_degrees[k], k))
    tables = {}
    for key, g in groups.items():
        angles = np.unique([a for _, a, _ in g["columns"]])
        n_r, n_t = len(g["specifiers"]), angles.size
        rho = np.full((n_r, n_t), np.nan)
        rad = np.full(n_r, np.nan)
        for i, a, k in g["columns"]:
            j = int(np.searchsorted(angles, a))
            rho[i, j] = densities[k]
            rad[i] = radii[k]
        if np.any(np.isnan(rho)):
            raise ValueError(
                f"density grid for {key} is not complete in radius x angle"
            )
        undefined = (rad <= 0.0) | ~np.isfinite(rad)
        rho[rho < 0.0] = 0.0  # undefined-value sentinel
        if np.all(undefined):
            rad = np.geomspace(
                1.0e-6, 1.0e-5, n_r
            )  # component absent; placeholder grid
            rho[:] = 0.0
        else:
            keep = ~undefined
            rad, rho = rad[keep], rho[keep]
            order = np.argsort(rad)
            rad, rho = rad[order], rho[order]
        tables[key] = DensityTable(key[0], key[1], rad, np.deg2rad(angles), rho)
    return tables
