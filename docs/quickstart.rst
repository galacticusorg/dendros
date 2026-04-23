Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install dendros

   # With optional pandas support:
   pip install 'dendros[pandas]'

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
``scale_factor``, and ``redshift``.

Listing properties
------------------

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       tbl = c.list_properties("Output1")  # by name
       tbl = c.list_properties(1)          # by 1-based integer index

Columns: ``name``, ``dtype``, ``shape``, ``description``, ``unitsInSI``.

Reading datasets
----------------

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       data = c.read("Output1", ["nodeData/basicMass"])
       # data["nodeData/basicMass"] is a numpy array

       # Custom labels via dict
       data = c.read(
           "Output1",
           {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
       )

Filtering
---------

Pass a boolean mask or integer index array as ``where``:

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       masses = c.read("Output1", ["nodeData/basicMass"])["nodeData/basicMass"]
       mask = masses > 1e12

       data = c.read(
           "Output1",
           {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
           where=mask,
       )

Tracing galaxy histories
------------------------

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

Multi-file collections search each file independently because
``nodeUniqueIDBranchTip`` is only unique *within* a single file (for MPI
outputs ``nodeUniqueIDBranchTip`` *is* unique across all files). By
default, if the same ID is found in more than one file at the same
output the call raises :class:`ValueError`; pass
``on_duplicate_file_match="warn"`` or ``"first"`` to keep the first
match instead.

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

