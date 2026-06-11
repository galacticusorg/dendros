Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install dendros

   # With optional pandas support:
   pip install 'dendros[pandas]'

   # With matplotlib for plotting analyses:
   pip install 'dendros[plot]'

   # Development version from GitHub:
   pip install git+https://github.com/galacticusorg/dendros.git

Opening files
-------------

.. code-block:: python

   from dendros import open_outputs

   # Single file
   c = open_outputs("galacticus.hdf5")

   # Auto-detect MPI-split outputs
   c = open_outputs("galacticus_MPI:0000.hdf5")

   # Explicit list or glob
   c = open_outputs(["rank0.hdf5", "rank1.hdf5"])
   c = open_outputs("run001/galacticus*.hdf5")

   # Lightcone mode
   c = open_outputs("lightcone.hdf5", output_root="Lightcone")

Use :class:`~dendros.Collection` as a context manager to ensure file handles
are closed automatically::

   with open_outputs("galacticus.hdf5") as c:
       tbl = c.list_outputs()

Checking completion status
--------------------------

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       c.validate_completion()           # raises RuntimeError if incomplete
       c.validate_completion(mode="warn")    # emit UserWarning instead
       c.validate_completion(mode="ignore")  # silent

Listing outputs
---------------

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       tbl = c.list_outputs()          # astropy Table
       df  = c.list_outputs(format="pandas")

The table contains columns: ``index``, ``name``, ``time``,
``scale_factor``, ``redshift``, and ``output_type``.  The ``output_type``
column reports the kind of output each group holds (``tree``, ``node``,
``snapshot``, or ``lightcone``); it is blank for older files that predate
the ``outputType`` attribute.

Listing properties
------------------

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       tbl = c.list_properties("Output1")  # by name
       tbl = c.list_properties(1)          # by 1-based integer index

Columns: ``name``, ``dtype``, ``shape``, ``description``, ``units``.  The
``units`` column shows a human-readable units description (e.g.
``"Solar masses"``), taken from the dataset's ``units`` attribute; it is
blank for dimensionless datasets.

Reading datasets
----------------

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       data = c.read("Output1", ["nodeData/basicMass"])
       # data["nodeData/basicMass"] is an astropy Quantity (in solar masses)

       # Custom labels via dict
       data = c.read(
           "Output1",
           {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
       )

Units and ``Quantity`` objects
------------------------------

By default, datasets that carry a units ``quantity`` string are returned as
:class:`astropy.units.Quantity` objects, so you can convert between units and
carry units through calculations:

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       mass = c.read("Output1", ["nodeData/basicMass"])["nodeData/basicMass"]

   mass.unit                 # Unit("solMass")
   mass.to("kg")             # convert to kilograms
   mass.value                # the underlying numpy array

Dimensionless datasets (those with an empty ``quantity``) are always returned
as plain :class:`numpy.ndarray` objects.  To disable the ``Quantity`` wrapping
entirely and get plain arrays for every dataset, pass ``as_quantity=False``:

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       data = c.read("Output1", ["nodeData/basicMass"], as_quantity=False)
       # data["nodeData/basicMass"] is a plain numpy array

Filtering
---------

Pass a boolean mask or integer index array as ``where``:

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       masses = c.read("Output1", ["nodeData/basicMass"])["nodeData/basicMass"]
       mask = masses.value > 1e12

       data = c.read(
           "Output1",
           {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
           where=mask,
       )

Tracing galaxy histories
------------------------

Tracing only makes sense for outputs in which a galaxy persists across cosmic
time — those of ``outputType`` ``tree`` or ``snapshot``.  In a ``lightcone``
output each galaxy is seen only once, and ``node`` outputs carry no persistent
cross-output identity, so :meth:`~dendros.Collection.trace_history` raises a
:class:`ValueError` if any chosen output is of type ``node`` or ``lightcone``.
Restrict to traceable outputs with the ``outputs=`` argument, or read those
outputs directly with :meth:`~dendros.Collection.read`.  Outputs from older
files that lack the ``outputType`` attribute are assumed traceable.

Given one or more ``nodeUniqueIDBranchTip`` values, dendros can assemble each
galaxy's full history across all outputs:

.. code-block:: python

   from dendros import open_outputs

   with open_outputs("galacticus.hdf5") as c:
       ids = [101, 104]   # branch-tip IDs of galaxies of interest
       hist = c.trace_history(
           ids,
           {"Mstar": "nodeData/diskMassStellar"},
       )

   # hist["Mstar"]             shape (2, n_outputs); NaN where absent
   # hist["time"]              shape (2, n_outputs); NaN where absent
   # hist["expansion_factor"]  shape (2, n_outputs)
   # hist["present"]           bool mask, shape (2, n_outputs)
   # hist["output_names"]      object array of output group names
   # hist["ids"]               int64 array, the normalized input

A 2-D per-galaxy property (e.g. a spectrum of shape ``(N_gals, n_bins)``) is
returned as a 3-D array of shape ``(n_galaxies, n_bins, n_outputs)`` — one
extra trailing axis for time. Each galaxy need not be present at every output
(it may have formed later or merged earlier), so history arrays are *ragged*
in time. Absent slots are filled with:

* ``NaN`` for floating-point properties (and for the ``time`` and
  ``expansion_factor`` arrays);
* the value of ``int_sentinel`` (default ``-1``) for integer properties;
* ``False`` for boolean properties.

The ``present`` mask is the canonical indicator of presence and should be
preferred to sentinel checks::

   import numpy as np
   mask = hist["present"][0]         # galaxy 0 presence across outputs
   times  = hist["time"][0][mask]
   masses = hist["Mstar"][0][mask]

Restrict to a subset of outputs with the ``outputs=`` argument (accepts a
``range``, a list of 1-based integers, or output group names)::

   hist = c.trace_history(ids, ["nodeData/basicMass"], outputs=range(1, 6))

Multi-file collections search each file independently. For arbitrary
user-provided file lists or globs, ``nodeUniqueIDBranchTip`` collisions are
possible across files, so by default if the same ID is found in more than
one file at the same output the call raises :class:`ValueError`; pass
``on_duplicate_file_match="warn"`` or ``"first"`` to keep the first
match instead. True Galacticus MPI-split outputs are a separate case:
there ``nodeUniqueIDBranchTip`` is expected to be unique across ranks/files
for a given output.

If ``nodeUniqueIDBranchTip`` was not included in the Galacticus run, the
function raises a :class:`KeyError` that points you at the missing output
property.

MPI outputs
-----------

Galacticus MPI runs produce files suffixed ``_MPI:NNNN``.  Dendros detects and
groups them automatically when you pass any single rank's filename or a glob::

   c = open_outputs("galacticus_MPI:0000.hdf5")  # auto-detects all peers

:meth:`~dendros.Collection.read` concatenates arrays from all ranks along
axis 0.

Star formation histories
------------------------

Star formation histories are output as lists of 2D :class:`numpy.ndarray` objects,
with one dimension being time, and the other metallicity. Dendros provides
functions to collapse (sum) over metallicity:

.. code-block:: python

   from dendros import sfh_collapse_metallicities, sfh_times

   with open_outputs("galacticus.hdf5") as c:
       sfh = c["Outputs/Output1/nodeData/diskStarFormationHistoryMass"]
       sfh_times = sfh_times(sfh)
       sfh_collapsed = sfh_collapse_metallicities(sfh)

Plotting analyses
-----------------

If a Galacticus run was configured to write reduced analysis results, the
HDF5 file will contain a top-level ``/analyses`` group with one subgroup per
analysis.  Dendros can list and plot every ``function1D`` analysis,
overlaying the model curve with the observational/target data when
present.  Requires the ``plot`` extra (``pip install 'dendros[plot]'``).

For MPI runs, the ``/analyses`` data is reduced over all ranks and is
identical in every rank's file, so dendros reads only the primary file.

.. code-block:: python

   from dendros import open_outputs

   with open_outputs("galacticus.hdf5") as c:
       # Tabulate available analyses (no matplotlib needed for this).
       print(c.list_analyses())

       # Plot every analysis; returns dict[name, matplotlib.figure.Figure].
       figs = c.plot_analyses()

       # Plot one analysis and also save to disk.
       figs = c.plot_analyses(
           name="stellarMassFunction",
           output_directory="figs",
           file_format="pdf",
       )

       # Hide the target overlay (model only).
       figs = c.plot_analyses(show_target=False)

To compare several models on the same figure, open them as a
:class:`~dendros.ModelCollection` with :func:`~dendros.open_models` and
pass the result to the module-level :func:`~dendros.plot_analyses`.  The
target overlay is shared across models so it is drawn only once; each
model contributes its own curve, labelled by the dict key (or by its
primary file stem when no dict is supplied).  MPI-split files are still
treated as a single model — only inter-model differences produce
additional curves.

.. code-block:: python

   from dendros import open_models, plot_analyses

   with open_models({"Fiducial": "fid.hdf5", "Variant": "var.hdf5"}) as m:
       figs = plot_analyses(m)

       # Or with default labels derived from filenames:
       # figs = plot_analyses(open_models(["fid.hdf5", "var.hdf5"]))

       # Or pass an explicit list with custom labels:
       # figs = plot_analyses(list(m.values()), labels=["A", "B"])

