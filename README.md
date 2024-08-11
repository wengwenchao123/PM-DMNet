This is a PyTorch implementation of PM-DMNet: Pattern-Matching Dynamic Memory Network for Traffic Prediction

## Table of Contents

* configs: training Configs and model configs for each dataset

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our model
  
* data: contains relevant datasets

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Data Preparation

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/file/d/1Q8boyeVNmZTz_HASN_57qd9wX1JZeGem/view?usp=sharing).

Unzip the downloaded dataset files into the `data` folder.

# Model Training
```bash
python run.py --datasets {DATASET_NAME} --type {MODEL_TYPE} --mode {MODE_NAME} 
```
Replace `{DATASET_NAME}` with one of datasets.

such as `python run.py --NYC-Taxi16 `

To run PM-DMNet with the desired configuration, set the `type` parameter accordingly:

- Set `type = "P"` to run PM-DMNet(P).
- Set `type = "R"` to run PM-DMNet(R).

This approach allows you to easily switch between the parallel prediction mode (PMP) and the recursive prediction mode (RMP) based on your specific requirements.

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the 'pre-trained' folder.
