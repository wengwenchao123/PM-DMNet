import os
import numpy as np
import pandas as pd
import h5py

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PeMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PeMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'METR-LA':
        data_path = os.path.join('./data/METR-LA/METR.h5')
        data = pd.read_hdf(data_path)
    elif dataset == 'BJ':
        data_path = os.path.join('./data/BJ/BJ500.csv')
        data = np.array(pd.read_csv(data_path, header=0, index_col=0))
    elif dataset == 'taxi':
        data_path = os.path.join('./data/taxi/taxi_data.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        # data = np.concatenate(rawdata, 1)
        data = np.stack(rawdata, -1)
    elif dataset == 'bike':
        data_path = os.path.join('./data/bike/bike_data.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "bike_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'NYCBike1':
        data_path = os.path.join('./data/NYCBike1/NYCBike1.npz')
        data = np.load(data_path,allow_pickle=True)['data'][:, :, :2].astype(float)
    elif dataset == 'NYCBike2':
        data_path = os.path.join('./data/NYCBike2/NYCBike2.npz')
        data = np.load(data_path,allow_pickle=True)['data'][:, :, :2].astype(float)
    elif dataset == 'NYCTaxi':
        data_path = os.path.join('./data/NYCTaxi/NYCTaxi.npz')
        data = np.load(data_path,allow_pickle=True)['data'][:, :, :2].astype(float)
    elif dataset == 'BJTaxi':
        data_path = os.path.join('./data/BJTaxi/BJTaxi.npz')
        data = np.load(data_path,allow_pickle=True)['data'][:, :, :2].astype(float)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

#
# data_path = os.path.join('../data/PeMS07/PEMS07.npz')
# data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
