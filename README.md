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


Dataset
---

Download the dataset for shapenet from: [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip)

Then to process the dataset, use the script as: `python3 src/dataset/data_process.py --dataroot <unzipped dataset path> --output <your data output path>`

This script will process the dataset and prepare it in the form of HDF5 files for each object separately. This will also apply the point encoding on the dataset.

Setup
---

To setup the required libraries for mesh processing, run the following command:
```
python3 setup.py build_ext --inplace
```

The following also need to be installed to run the code properly:
```
pip3 install --user pytorch-lightning efficientnet-pytorch pykdtree
```

Training
---

To train the model, use the following command:
```
$ python3 src/train.py --help
usage: train.py [-h] [--cdim CDIM] [--hdim HDIM] [--pdim PDIM] [--data_root DATA_ROOT] [--batch_size BATCH_SIZE] [--output_path OUTPUT_PATH] [--exp_name EXP_NAME] [--encoder ENCODER] [--decoder DECODER]

Argument parser for training the model

optional arguments:
  -h, --help            show this help message and exit
  --cdim CDIM           feature dimension
  --hdim HDIM           hidden size for decoder
  --pdim PDIM           points input size for decoder
  --data_root DATA_ROOT
                        location of the parsed and processed dataset
  --batch_size BATCH_SIZE
                        Training batch size
  --output_path OUTPUT_PATH
                        Model saving and checkpoint paths
  --exp_name EXP_NAME   Name of the experiment. Artifacts will be created with this name
  --encoder ENCODER     Name of the Encoder architecture to use
  --decoder DECODER     Name of the decoder architecture to use
```

Fill the values accordingly for the configuration and the model shall start training. We can also make use of mixed precision training via pytorch lightning. To do this, edit the `src/trainer.py` script.

To view the training progress, run tensorboard in your experiment directory `tensorboard --logdir=<experiment directory>`


Evaluation
----
To run evaluation on the test set (selective objects: roughly 500 for now. Change it in the script for more), use the following script:

```
$ python3 run_evals.py --help
usage: run_evals.py [-h] [--cdim CDIM] [--hdim HDIM] [--pdim PDIM] [--data_root DATA_ROOT] [--batch_size BATCH_SIZE] [--output_path OUTPUT_PATH] [--exp_name EXP_NAME] [--encoder ENCODER] [--decoder DECODER]
                    [--checkpoint CHECKPOINT]

Argument parser for training the model

optional arguments:
  -h, --help            show this help message and exit
  --cdim CDIM           feature dimension
  --hdim HDIM           hidden size for decoder
  --pdim PDIM           points input size for decoder
  --data_root DATA_ROOT
                        location of the parsed and processed dataset
  --batch_size BATCH_SIZE
                        Training batch size
  --output_path OUTPUT_PATH
                        Model saving and checkpoint paths
  --exp_name EXP_NAME   Name of the experiment. Artifacts will be created with this name
  --encoder ENCODER     Name of the Encoder architecture to use
  --decoder DECODER     Name of the decoder architecture to use
  --checkpoint CHECKPOINT
                        Checkpoint Path
```


Visualization
---

To generate 3D models and meshes, use `jupyter` notebook or lab environment, and run `check_model.ipynb`. Keep the config flags same as evaluation and tweak the save path to see results.