# Occupancy Networks for Single View Reconstruction

__Team__: Noisy Pixels

Metrics Functionality uses pykdtree library. pykdtree is a kd-tree implementation for fast nearest neighbour search in Python.The implementation is based on scipy.spatial.cKDTree and libANN by combining the best features from both and focus on implementation efficiency.

Installation
------------

.. code-block:: bash

    $ cd <pykdtree_dir>
    $ python setup.py install
    
Code structure:
```
- resources
  - propossal.pdf
  - mid_eval.pdf
- src
  - dataset
    - __init__.py
    - dataloader.py
  - models
    - __init__.py
    - encoder.py
    - decoder.py
  - viz
    - visualization.py
  - train.py
  - test.py
  - run.py
- demo.py
- README.md
- proposal.pdf
```
