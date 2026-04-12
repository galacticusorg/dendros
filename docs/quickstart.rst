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
       data = c.read("Output1", ["nodeData/haloMass"])
       # data["nodeData/haloMass"] is a numpy array

       # Custom labels via dict
       data = c.read(
           "Output1",
           {"Mhalo": "nodeData/haloMass", "Mstar": "nodeData/stellarMass"},
       )

Filtering
---------

Pass a boolean mask or integer index array as ``where``:

.. code-block:: python

   with open_outputs("galacticus.hdf5") as c:
       masses = c.read("Output1", ["nodeData/haloMass"])["nodeData/haloMass"]
       mask = masses > 1e12

       data = c.read(
           "Output1",
           {"Mhalo": "nodeData/haloMass", "Mstar": "nodeData/stellarMass"},
           where=mask,
       )

MPI outputs
-----------

Galacticus MPI runs produce files suffixed ``_MPI:NNNN``.  Dendros detects and
groups them automatically when you pass any single rank's filename or a glob::

   c = open_outputs("galacticus_MPI:0000.hdf5")  # auto-detects all peers

:meth:`~dendros.Collection.read` concatenates arrays from all ranks along
axis 0.
