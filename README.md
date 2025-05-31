## [TITS2025] Pattern-Matching Dynamic Memory Network for Dual-Mode Traffic Prediction

This is a PyTorch implementation of **[Pattern-Matching Dynamic Memory Network for Dual-Mode Traffic Prediction](https://ieeexplore.ieee.org/document/10992276)**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pattern-matching-dynamic-memory-network-for-1/traffic-prediction-on-pemsd7-l)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7-l?p=pattern-matching-dynamic-memory-network-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pattern-matching-dynamic-memory-network-for-1/traffic-prediction-on-pemsd8)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd8?p=pattern-matching-dynamic-memory-network-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pattern-matching-dynamic-memory-network-for-1/traffic-prediction-on-pemsd7)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7?p=pattern-matching-dynamic-memory-network-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pattern-matching-dynamic-memory-network-for-1/traffic-prediction-on-pemsd7-m)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7-m?p=pattern-matching-dynamic-memory-network-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pattern-matching-dynamic-memory-network-for-1/traffic-prediction-on-pemsd4)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd4?p=pattern-matching-dynamic-memory-network-for-1)

## Update
 (2025/4/22)
* Good news! This paper is accepted by IEEE Transactions on Intelligent Transportation Systems.

  
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

such as `python run.py --dataset NYC-Taxi16 `

To run PM-DMNet with the desired configuration, set the `type` parameter accordingly:

- Set `type P` to run PM-DMNet(P).
- Set `type R` to run PM-DMNet(R).

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the 'pre-trained' folder.

Here is an example of how to run the script using the specified parameters:
```bash
python run.py --dataset PEMSD8 --type P --mode train
```

## Cite

If you find the paper useful, please cite as following:

```
@article{weng2025pattern,
  title={Pattern-Matching Dynamic Memory Network for Dual-Mode Traffic Prediction},
  author={Weng, Wenchao and Wu, Mei and Jiang, Hanyu and Kong, Wanzeng and Kong, Xiangjie and Xia, Feng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```

## More Related Works

- [[Neural Networks] RGDAN: A random graph diffusion attention network for traffic prediction](https://doi.org/10.1016/j.neunet.2023.106093)
  
- [[Pattern Recognition] A Decomposition Dynamic Graph Convolutional Recurrent Network for Traffic Forecasting](https://www.sciencedirect.com/science/article/pii/S0031320323003710)

- [[Neural Networks] PDG2Seq: Periodic Dynamic Graph to Sequence model for Traffic Flow Prediction](https://doi.org/10.1016/j.neunet.2024.106941)
  

