import numpy as np
import torch
from torch.utils.data import Dataset

"""
Timeseries sampler Dataset
版本说明：目前实现的为简易版本，只实现了基础功能
Todo:
1. 数据长度不足的补足处理
2. 采样策略的优化
用于从时间序列数据中采样出固定长度的样本窗口（insample）和对应的输出窗口（outsample）
"""
class TimeseriesDataset(Dataset):
    def __init__(self,
                 timeseries: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 ):
        self.timeseries = [ts for ts in timeseries]
        self.insample_size = insample_size
        self.outsample_size = outsample_size

    def __len__(self):
        return len(self.timeseries)

    def __getitem__(self, idx):
        #data length 为insample_size + outsample_size，当前的data包含了insample和outsample,后续需要再simulator中进行分离    
        #Todo : 数据长度不足的补足处理
        data = self.timeseries[idx][-self.insample_size-self.outsample_size:]
        
        return torch.FloatTensor(data)
        