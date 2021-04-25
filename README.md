# Occupancy Networks for Single View Reconstruction

__Team__: Noisy Pixels

__Team Members__:<br>
Shubham Dokania _(2020701016)_<br>
Shanthika Shankar Naik _(2020701013)_ <br>
Sai Amrit Patnaik _(2020701026)_ <br>
Madhvi Panchal _(2019201061)_ <br>

__Assigned TA__: Meher Shashwat Nigam <br><br>

This project is undertaken as a part of the Computer Vision coursework at IIIT Hyderabad in Spring semester 2021. The paper implemented in this project is: [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mescheder_Occupancy_Networks_Learning_3D_Reconstruction_in_Function_Space_CVPR_2019_paper.pdf) by _Mescheder et. al._

The approach focuses on implicit learning of 3D surface as a continuous decision boundary of a non-linear classifier. Occupancy networks implicitly represent the 3D surface as the continuous decision boundary of a deep neural network classifier. The details of the implementation have been outlined in the project report and the proposal document which can be found [here](./resources/proposal.pdf).

The following sections outline how to run the demo and some examples of the expected output from running the mentioned scripts.

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

In the above structure, the source code for the whole implementation can be found in the `src` directory. The scripts each contain a description of the functions/classes implemented and provide a wrapper to experiment with the flow of the program.

Metrics Functionality uses pykdtree library. pykdtree is a kd-tree implementation for fast nearest neighbour search in Python.The implementation is based on scipy.spatial.cKDTree and libANN by combining the best features from both and focus on implementation efficiency.

Installation
------------

    $ cd <pykdtree_dir>
    $ python setup.py install
