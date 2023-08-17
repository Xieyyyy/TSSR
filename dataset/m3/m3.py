"""
M3 Dataset processing
"""
import os
from dataclasses import dataclass

import fire
import numpy as np
import pandas as pd

#DATASETS_PATH 为上一级目录下的datasets文件夹
DATASETS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(DATASETS_PATH, 'm3')
DATASET_PATH = os.path.join(DATASET_PATH, 'datafiles')
DATASET_FILE_PATH = os.path.join(DATASET_PATH, 'M3C.xls')


TRAINING_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npy')
TEST_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'test.npy')
IDS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'ids.npy')
GROUPS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'groups.npy')
HORIZONS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'horizons.npy')


@dataclass()
class M3Meta:
    seasonal_patterns = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']
    horizons = [6, 8, 18, 8] #预测的步长，四个sheetM3Year、M3Quart、M3Month、M3Other中的数据的预测步长分别为6、8、18、8
    frequency = [1, 4, 12, 1]
    horizons_map = {
        'M3Year': 6,
        'M3Quart': 8,
        'M3Month': 18,
        'M3Other': 8
    }


@dataclass()
class M3Dataset:
    ids: np.ndarray
    groups: np.ndarray #四个sheet中的数据group标记分别M3Year、M3Quart、M3Month、M3Other
    horizons: np.ndarray
    values: np.ndarray

    ## 读取数据集，返回M3Dataset对象，包含ids、groups、horizons、values
    ## 当training为True时，返回的values为训练集的数据，否则为测试集的数据
    @staticmethod
    def load(training: bool = True) -> 'M3Dataset':
        values_file = TRAINING_SET_CACHE_FILE_PATH if training else TEST_SET_CACHE_FILE_PATH
        return M3Dataset(ids=np.load(IDS_CACHE_FILE_PATH, allow_pickle=True),
                         groups=np.load(GROUPS_CACHE_FILE_PATH, allow_pickle=True),
                         horizons=np.load(HORIZONS_CACHE_FILE_PATH, allow_pickle=True),
                         values=np.load(values_file, allow_pickle=True))

    @staticmethod
    def process_data() -> None:
        """
        初步处理M3的数据，存储为np.array格式
        """
        ids = []
        groups = []
        horizons = []
        training_values = []
        test_values = []
        #horizon为预测的步长，按照预测的步长分割数据文件，如果horizon为6，数据总长度为20，那么训练集将为14，测试集为6
        for sp in M3Meta.seasonal_patterns:
            horizon = M3Meta.horizons_map[sp] 
            dataset = pd.read_excel(DATASET_FILE_PATH, sheet_name=sp) #读取excel文件，分为四个sheet，分别为年度、季度、月度、其他
            ids.extend(dataset[['Series']].values[:, 0]) #Series为数据的id，唯一标识
            horizons.extend(dataset['NF'].values) #NF为预测的步长
            groups.extend(np.array([sp] * len(dataset)))
            #前七列为数据的描述信息，后面的列为数据
            training_values.extend([ts[~np.isnan(ts)][:-horizon] for ts in dataset[dataset.columns[6:]].values])
            test_values.extend([ts[~np.isnan(ts)][-horizon:] for ts in dataset[dataset.columns[6:]].values])

        np.save(IDS_CACHE_FILE_PATH, ids, allow_pickle=True)
        np.save(GROUPS_CACHE_FILE_PATH, groups, allow_pickle=True)
        np.save(HORIZONS_CACHE_FILE_PATH, horizons, allow_pickle=True)
        np.save(TRAINING_SET_CACHE_FILE_PATH, training_values, allow_pickle=True)
        np.save(TEST_SET_CACHE_FILE_PATH, test_values, allow_pickle=True)


if __name__ == '__main__':

    M3Dataset.process_data()
