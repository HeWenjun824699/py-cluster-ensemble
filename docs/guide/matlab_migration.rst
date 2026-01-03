================
MATLAB Migration
================

A primary design goal of PCE is to bridge the gap between MATLAB-based research and the Python ecosystem.

Handling .mat Files
-------------------

PCE provides a robust interface to interact with MATLAB data files. It automatically detects and supports both standard MATLAB formats and the **v7.3 (HDF5-based)** format.

* **Feature Matrices**: Use ``io.load_mat_X_Y`` to load datasets containing feature matrices (X) and ground truth labels (Y).
* **Precomputed Partitions**: Use ``io.load_mat_BPs_Y`` to import base partitions (BPs) generated in MATLAB for further ensemble processing in Python.

Automatic Index Correction
--------------------------

One of the most frequent issues when migrating from MATLAB to Python is the difference in indexing (1-based vs. 0-based).

* **The Problem**: MATLAB identifies clusters starting from 1, while Python's Scikit-learn and NumPy use 0-based indexing.
* **The Solution**: PCE's loading functions include a ``fix_matlab_index`` parameter (defaulting to ``True``). If the toolkit detects that the minimum cluster label in a loaded ``BPs`` matrix is 1, it automatically subtracts 1 from all entries to ensure compatibility with Python algorithms.

Data Persistence
----------------

PCE ensures your research results can be sent back to MATLAB users effortlessly. The ``io.save_results_mat`` and ``io.save_base_mat`` functions preserve the data structure and ensure labels are reshaped into the standard MATLAB column vector format (:math:`N \times 1`).
