import argparse
import time

import numpy as np
import torch

from data_provider.data_factory import data_provider
from engine import Engine

from torch.utils.tensorboard import SummaryWriter

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

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)
torch.backends.cudnn.benchmark = True


def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def record_info(info, file_dir):
    with open(file_dir, 'a') as f:
        f.writelines(info + "\n")


def main():
    train_data, train_loader = get_data(args=args, flag='train')
    vali_data, vali_loader = get_data(args=args, flag='val')
    test_data, test_loader = get_data(args=args, flag='test')

    if args.recording:
        # record_info(str(args), "./records/" + args.tag)
        sw = SummaryWriter(comment=args.tag)

    engine = Engine(args)

    print("start training...")
    train_time = []
    for epoch_num in range(args.epoch + 1):
        t1 = time.time()
        train_loss = 0
        train_n_samples = 0
        train_maes, train_mses, train_corrs, test_maes, test_mses, test_corrs = [], [], [], [], [], []
        for iter, (data, _, _, _) in enumerate(train_loader):
            train_data = data[..., args.used_dimension].float()
            all_eqs, all_times, test_data, loss, mae, mse, corr = engine.simulate(train_data)
            train_maes.append(mae)
            train_mses.append(mse)
            train_corrs.append(corr)
            train_loss += loss
            train_n_samples += 1


            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MSE: {:.4f}, Train CORR: {:.4f}'
            print(log.format(iter, train_loss / train_n_samples, mae, mse, corr), flush=True)

        torch.cuda.empty_cache()
        print("eval...")
        for iter, (data, _, _, _) in enumerate(test_loader):
            train_data = data[..., args.used_dimension].float()
            mae, mse, corr, all_eqs, test_data = engine.eval(train_data)
            test_maes.append(mae)
            test_mses.append(mse)
            test_corrs.append(corr)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Test MAE: {:.4f}, Test MSE: {:.4f}, Test CORR: {:.4f}, ' \
              'Training Time: {:.4f}/epoch'
        print(log.format(epoch_num, train_loss / train_n_samples, sum(test_maes) / len(test_maes),
                         sum(test_mses) / len(test_mses), sum(test_corrs) / len(test_corrs), time.time() - t1),
              flush=True)

        if args.recording:
            sw.add_scalar('Loss/train', train_loss / train_n_samples, global_step=epoch_num)
            sw.add_scalar('MAE/valid', sum(test_maes) / len(test_maes), global_step=epoch_num)
            sw.add_scalar('MSE/valid', sum(test_mses) / len(test_mses), global_step=epoch_num)
            sw.add_scalar('RSE/valid', sum(test_corrs) / len(test_corrs), global_step=epoch_num)


if __name__ == '__main__':
    main()
