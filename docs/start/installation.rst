============
Installation
============

PCE requires Python 3.10 or higher.

Install via pip
---------------

The simplest way to install ``py-cluster-ensemble`` is using ``pip``:

.. code-block:: bash

   pip install py-cluster-ensemble

Dependencies
------------

PCE relies on the following core libraries which will be installed automatically:

* **NumPy/SciPy**: For core numerical operations and matrix manipulations.
* **Scikit-learn**: For base clustering implementations and dimensionality reduction.
* **Pandas**: For experiment logging and result persistence in CSV/Excel formats[cite: 3, 4].
* **Matplotlib/Seaborn**: For generating paper-quality visualizations[cite: 9].
* **H5py**: To ensure compatibility with MATLAB v7.3 ``.mat`` files[cite: 3].