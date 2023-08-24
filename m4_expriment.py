# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Experiment
"""
import numpy as np
from data_provider.data_factory import data_provider

from TimeseriesDataset import TimeseriesDataset

from dataset.m4.m4 import M4Dataset, M4Meta

#import engin
from engine import Engine
#import DataLoader
from torch.utils.data import DataLoader
from data_provider.utils.tools import group_values, get_data
from m4_args import M4Args

# def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
#     """
#     Filter values array by group indices and clean it from NaNs.
#     """
#     return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])

# def get_data(in_data):
#     data_set, data_loader = data_provider(in_data)
#     return data_set, data_loader

class M4Experiment:
    """
    M3 Experiment : 执行M4的实验
    lookback: 1 设置默认的回溯窗口为1，当前简易版本horizon和lookback相同    
    """
    @staticmethod
    def run(lookback=1):
        batch_size = 1
        engine  = Engine(args=M4Args.get_args()) #初始化引擎参数    
        dataset = M4Dataset.load_data() #加载数据集(剔除了测试部分的6位)

        for seasonal_pattern in ['Daily']: ##选择M3Year先测试一下，M3Meta.seasonal_patterns
            horizon = 48
            #input_size = lookback * horizon   # 当前简易版本中输入和输出都是horizon长度,后续优化再利用input_size

            # Training data
            #training_values = group_values(dataset.training_values, dataset.groups, seasonal_pattern)

            #Training Dataset
            training_data = TimeseriesDataset(timeseries=dataset.training_values,
                                            insample_size=horizon,
                                            outsample_size=horizon) 
            
            #测试先使用简易版本 ,后续利用data_provider读取数据集
            train_loader = DataLoader(training_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=True)
            
            # model training
            print("start training...")
            for epoch in range(10):
                train_loss = 0
                train_n_samples = 0
                train_maes, train_mses, train_corrs, test_maes, test_mses, test_corrs = [], [], [], [], [], []
                for iter, data in enumerate(train_loader):
                    train_data = data.float() #[..., args.used_dimension]
                    all_eqs, all_times, test_data, loss, mae, mse, corr = engine.simulate(train_data)
                    print("epoch: {}, iter: {}, loss: {}, mae: {}, mse: {}, corr: {}".format(epoch, iter, loss, mae, mse, corr))
                    train_maes.append(mae)
                    train_mses.append(mse)
                    train_corrs.append(corr)
                    train_loss += loss
                    train_n_samples += 1


if __name__ == '__main__':
    M4Experiment.run()
