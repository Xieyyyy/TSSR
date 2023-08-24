from dataclasses import dataclass
import os
import numpy as np

import pandas as pd


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
DATASETS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(DATASETS_PATH, 'm4')
DATASET_PATH = os.path.join(DATASET_PATH, 'datafiles')
file_names = ['Daily-train.csv', 'Hourly-train.csv', 'Monthly-train.csv', 'Quarterly-train.csv',
                  'Weekly-train.csv', 'Yearly-train.csv']
DATASET_FILE_PATH = os.path.join(DATASET_PATH, file_names[0]) #先利用Daily-train.csv文件进行测试

TRAINING_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npy')
IDS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'ids.npy')
GROUPS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'groups.npy')
HORIZONS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'horizons.npy')

@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    training_values: np.ndarray

    @staticmethod
    def load_data() -> None:
        """
        初步处理M3的数据，存储为np.array格式
        """
        ids = []
        groups = []
        training_values = []
        #horizon为预测的步长，按照预测的步长分割数据文件，如果horizon为6，数据总长度为20，那么训练集将为14，测试集为6
        # for sp in M4Meta.seasonal_patterns:
        # horizon = 48
        dataset = pd.read_csv(DATASET_FILE_PATH) #读取csv文件
        ids.extend(dataset[['V1']].values[:, 0]) #Series为数据的id，唯一标识
        groups.extend(['Daily'] * len(dataset)) #M4Meta.seasonal_patterns[4]为Daily
        training_values.extend([ts[~np.isnan(ts)] for ts in dataset[dataset.columns[1:]].values])

        return M4Dataset(ids=ids, training_values=training_values, groups=groups)

if __name__ == '__main__':

    M4Dataset.load_data()
