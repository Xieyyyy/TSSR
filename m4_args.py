import argparse
import numpy as np

# a simple class provide args for M3Experiment
# seq_in = 6， seq_out = 6 临时测试的设置，后续改为48
class M4Args:   
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
        parser.add_argument("--device", type=str, default="cpu")
        # -- data processing
        parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='illness/national_illness.csv', help='data file')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding,'
                                'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                                'you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                                'S:univariate predict univariate, '
                                'MS:multivariate predict univariate')

        # -- model argments
        parser.add_argument('--used_dimension', type=int, default=1)
        parser.add_argument('--symbolic_lib', type=str, default="elec_small")
        parser.add_argument('--max_len', type=int, default=20)
        parser.add_argument('--max_module_init', type=int, default=10)
        parser.add_argument('--num_transplant', type=int, default=2)
        parser.add_argument('--num_runs', type=int, default=1)
        parser.add_argument('--eta', type=float, default=1)
        parser.add_argument('--num_aug', type=int, default=0)
        parser.add_argument('--exploration_rate', type=float, default=1 / np.sqrt(2))
        parser.add_argument('--transplant_step', type=int, default=1000)
        parser.add_argument('--norm_threshold', type=float, default=1e-5)

        # -- training
        parser.add_argument("--seed", type=int, default=42, help='random seed')
        parser.add_argument("--epoch", type=int, default=50, help='epoch')
        parser.add_argument("--epoch_train", type=int, default=10, help='epoch')
        parser.add_argument("--seq_in", type=int, default=48, help='length of input seq')
        parser.add_argument("--seq_out", type=int, default=48, help='length of output seq')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument("--batch_size", type=int, default=1, help='default')
        parser.add_argument("--train_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-6, help='learning rate')
        parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
        parser.add_argument("--clip", type=float, default=5., help='gradient clip')
        parser.add_argument("--lr_decay", type=float, default=1)

        # -- analysis
        parser.add_argument("--recording", action="store_true")
        parser.add_argument("--tag", type=str, default="solar_baseline", help='')
        args = parser.parse_args()  # args init
        return args