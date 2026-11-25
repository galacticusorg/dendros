"""Visualizations of Galacticus models.

Images are built from the model's own description of each galaxy: the density of
each component tabulated by the ``densityProfile`` node property extractor (on a grid
of radius and polar angle), the luminosity of each component in each band, and the
gas mass and metallicity, from which the dust follows.

Quick start::

    from dendros import open_outputs
    from dendros.viz import galaxy_scene, render_galaxy, compose_rgb

    with open_outputs("galaxies.hdf5") as c:
        scene = galaxy_scene(c, "Output1", index)
    images = render_galaxy(scene, inclination=65.0)
    rgb = compose_rgb(images)

Requires scipy and astropy (``pip install 'dendros[viz]'``).
"""

from ._dynamics import (
    DiskDynamics,
    RotationCurve,
    disk_dynamics,
    dynamical_spiral_texture,
    rotation_curves,
)
from ._galaxy import GalaxyScene, GasComponent, StellarComponent, galaxy_scene
from ._profiles import DensityTable, density_tables
from ._render import (
    DustModel,
    compose_rgb,
    render_galaxy,
    rotation_matrix,
    surface_brightness_magnitudes,
)
from ._lightcone import (
    LightconeField,
    LightconeGalaxy,
    add_noise,
    lightcone_field,
    render_lightcone,
)
from ._orbits import (
    SubhaloOrbits,
    compose_orbit_animation,
    render_orbits,
    subhalo_orbits,
)
from ._story import (
    StoryFrame,
    compose_story_animation,
    life_story_figure,
    load_story_frames,
    presentation_render,
    render_main_branch,
)
from ._texture import SpiralTexture
from ._tracks import Histories, draw_tracks, main_branch_histories, render_tracks
from ._tree import (
    Branch,
    GalaxyMergerTree,
    RiverLayout,
    draw_river_tree,
    galaxy_merger_tree,
    halo_growth_rate,
    halo_merger_tree,
    main_branch_rows,
    river_colormap,
    radial_layout,
    render_radial,
    render_river,
    river_layout,
    specific_star_formation_rate,
)

__all__ = [
    "Branch",
    "DensityTable",
    "DiskDynamics",
    "DustModel",
    "GalaxyMergerTree",
    "GalaxyScene",
    "GasComponent",
    "Histories",
    "LightconeField",
    "LightconeGalaxy",
    "RiverLayout",
    "RotationCurve",
    "SpiralTexture",
    "StellarComponent",
    "StoryFrame",
    "SubhaloOrbits",
    "add_noise",
    "compose_orbit_animation",
    "compose_rgb",
    "compose_story_animation",
    "density_tables",
    "disk_dynamics",
    "dynamical_spiral_texture",
    "draw_river_tree",
    "draw_tracks",
    "galaxy_merger_tree",
    "galaxy_scene",
    "halo_growth_rate",
    "halo_merger_tree",
    "life_story_figure",
    "lightcone_field",
    "load_story_frames",
    "main_branch_histories",
    "main_branch_rows",
    "presentation_render",
    "radial_layout",
    "render_galaxy",
    "render_lightcone",
    "render_main_branch",
    "render_orbits",
    "render_radial",
    "render_river",
    "render_tracks",
    "river_colormap",
    "river_layout",
    "rotation_matrix",
    "rotation_curves",
    "specific_star_formation_rate",
    "subhalo_orbits",
    "surface_brightness_magnitudes",
]
