"""Multivariate-normal fit to post-burn chains and the reparameterization config writer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np

from ._chains import ChainSet


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MVNFit:
    """Multivariate-normal fit to post-burn samples.

    Attributes
    ----------
    mean:
        ``(n_params,)`` sample mean.
    covariance:
        ``(n_params, n_params)`` sample covariance, symmetrised.
    cholesky:
        ``(n_params, n_params)`` lower-triangular Cholesky factor of
        :attr:`covariance`.  Satisfies ``L @ L.T == covariance``.
    parameter_names:
        Parameter names along the axes.

    Methods
    -------
    write_reparameterization_config:
        Emit a Galacticus-style XML config that re-parameterizes the active
        parameters in terms of independent unit-normal meta parameters.
    """

    mean: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    parameter_names: Tuple[str, ...]

    def write_reparameterization_config(
        self,
        out_path: Union[str, "Path"],
        *,
        n_sigma: float = 5.0,
        perturber_scale: float = 1.0e-5,
    ) -> Path:
        """Write a Galacticus reparameterization XML config.

        For an *n*-parameter MVN fit with mean :math:`\\mu` and Cholesky
        factor :math:`L`, the emitted config declares *n* active
        ``metaParameter{i}`` parameters with truncated unit-normal priors
        (limits :math:`\\pm n_\\sigma`), and *n* derived parameters expressing
        the original active parameters as

        .. math::

            x_i = \\mu_i + \\sum_j L_{ij} \\, m_j .

        Re-running the MCMC against this config samples in coordinates where
        the posterior is approximately spherical.

        Parameters
        ----------
        out_path:
            Destination path.
        n_sigma:
            Truncation half-width for the meta-parameter priors, in units of
            their (unit) standard deviation.  Defaults to ``5.0``.
        perturber_scale:
            Cauchy ``scale`` of the per-meta-parameter perturber.  Defaults to
            ``1.0e-5`` to match the Galacticus reference.

        Returns
        -------
        pathlib.Path
            Resolved path of the written file.
        """
        if n_sigma <= 0:
            raise ValueError(f"n_sigma must be positive; got {n_sigma!r}")
        if perturber_scale <= 0:
            raise ValueError(
                f"perturber_scale must be positive; got {perturber_scale!r}"
            )

        n = len(self.parameter_names)
        if n == 0:
            raise ValueError("MVNFit has no parameters to write.")

        root = ET.Element("parameters")

        for i in range(n):
            mp = ET.SubElement(root, "modelParameter", value="active")
            ET.SubElement(mp, "name", value=f"metaParameter{i}")
            prior = ET.SubElement(mp, "distributionFunction1DPrior", value="normal")
            ET.SubElement(prior, "mean", value="0.0")
            ET.SubElement(prior, "variance", value="1.0")
            ET.SubElement(prior, "limitLower", value=f"{-n_sigma:.6g}")
            ET.SubElement(prior, "limitUpper", value=f"{n_sigma:.6g}")
            ET.SubElement(mp, "operatorUnaryMapper", value="identity")
            pert = ET.SubElement(
                mp, "distributionFunction1DPerturber", value="cauchy"
            )
            ET.SubElement(pert, "median", value="0.0")
            ET.SubElement(pert, "scale", value=f"{perturber_scale:.6g}")

        for i, name in enumerate(self.parameter_names):
            mp = ET.SubElement(root, "modelParameter", value="derived")
            ET.SubElement(mp, "name", value=name)
            terms = [f"{self.mean[i]:.16g}"]
            for j in range(n):
                coef = float(self.cholesky[i, j])
                if coef == 0.0:
                    continue
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign}{abs(coef):.16g}*%[metaParameter{j}]")
            definition = "".join(terms)
            ET.SubElement(mp, "definition", value=definition)

        tree = ET.ElementTree(root)
        _indent_inplace(tree.getroot(), level=0, space="  ")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return out.resolve()


def _indent_inplace(elem: ET.Element, level: int = 0, space: str = "  ") -> None:
    """In-place pretty-indenter equivalent to ``xml.etree.ElementTree.indent``.

    Provided for Python 3.8, where ``ET.indent`` is not available.
    """
    i = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + space
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent_inplace(child, level + 1, space)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def multivariate_normal_fit(
    chains: ChainSet,
    *,
    post_burn: Optional[int] = None,
    drop_chains: Sequence[int] = (),
) -> MVNFit:
    """Fit a multivariate normal to the post-burn concatenated chain.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    post_burn:
        Number of leading rows to skip per chain.  ``None`` triggers
        automatic detection via :func:`gelman_rubin` /
        :func:`convergence_step`.
    drop_chains:
        ``chain_index`` values to exclude.

    Returns
    -------
    MVNFit

    Raises
    ------
    ValueError
        If fewer than ``n_params + 1`` post-burn samples remain (so that the
        sample covariance is rank-deficient).
    np.linalg.LinAlgError
        If the sample covariance is not positive-definite (which can happen
        for parameters that are degenerate post-burn).  Drop the offending
        parameter or supply more samples.
    """
    from ._convergence import _resolve_post_burn

    burn = _resolve_post_burn(chains, post_burn)
    drop = set(int(i) for i in drop_chains)

    parts = [
        c.state[burn:]
        for c in chains
        if c.chain_index not in drop and c.n_steps > burn
    ]
    if not parts:
        raise ValueError(
            "No post-burn samples available — every chain was dropped or "
            "shorter than post_burn."
        )
    samples = np.concatenate(parts, axis=0)

    n_samples, n_params = samples.shape
    if n_samples < n_params + 1:
        raise ValueError(
            f"Need at least n_params + 1 = {n_params + 1} post-burn samples "
            f"to fit a multivariate normal; got {n_samples}."
        )

    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    L = np.linalg.cholesky(cov)

    return MVNFit(
        mean=mean,
        covariance=cov,
        cholesky=L,
        parameter_names=chains.config.parameter_names,
    )
