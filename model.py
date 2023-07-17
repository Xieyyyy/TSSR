import time

import numpy as np
import torch.nn as nn

import score
from MCTSblock import MCTSPlayer


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.max_len = args.max_len
        self.max_module_init = args.max_module_init
        self.num_transplant = args.num_transplant
        self.num_runs = args.num_runs
        self.eta = args.eta
        self.num_aug = args.num_aug
        self.score_with_est = score.score_with_est
        self.exploration_rate = args.exploration_rate
        self.transplant_step = args.transplant_step
        self.norm_threshold = args.norm_threshold

    def forward(self, X, y):

        count_success = True
        assert X.size(0) == 1
        X = X.squeeze(0)
        y = y.squeeze(0)

        time_idx = np.arange(X.size(0) + y.shape[0])
        input_data = np.vstack([time_idx[:X.size(0)], X])

        supervision_data = np.vstack([time_idx, np.concatenate([X, y])])

        num_success = 0
        all_times = []
        all_eqs = []

        for i_test in range(self.num_runs):
            best_solution, discovery_time, test_score = self.run_discovery(i_test, supervision_data)
            if discovery_time != 0:
                all_times.append(discovery_time)
                num_success += 1
            all_eqs.append(score.simplify_eq(best_solution[0]))

            print('best solution: {}'.format(score.simplify_eq(best_solution[0])))
            print('test score: {}'.format(test_score))
            print()

        success_rate = num_success / self.num_runs
        if count_success:
            print('success rate :', success_rate)

        return all_eqs, success_rate, all_times, supervision_data

    def run_discovery(self, i_test, supervision_data):
        best_solution = ('nothing', 0)

        exploration_rate = self.exploration_rate  # 设置探索率
        max_module = self.max_module_init  # 设置最大模块
        reward_his = []  # 初始化奖励历史
        best_modules = []  # 初始化最佳模块
        aug_grammars = []  # 初始化增强语法

        start_time = time.time()  # 记录开始时间
        discovery_time = 0  # 初始化发现时间
        module_grow_step = (self.max_len - self.max_module_init) / self.num_transplant

        for i_itr in range(self.num_transplant):
            mcts_block = MCTSPlayer(max_aug=self.num_aug,
                                    max_module=self.max_module_init, scale=0, max_len=self.max_len,
                                    lib_name="elec_small", exploration_rate=1 / np.sqrt(2), num_episodes=1000,
                                    num_play=50)

            _, current_solution, good_modules = mcts_block.run(supervision_data)

            end_time = time.time() - start_time  # 计算运行时间

            if not best_modules:
                best_modules = good_modules
            else:
                best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

            aug_grammars = [x[0] for x in best_modules[-self.num_aug:]]

            reward_his.append(best_solution[1])

            if current_solution[1] > best_solution[1]:
                best_solution = current_solution

            max_module += module_grow_step
            exploration_rate *= 5

            test_score = \
                self.score_with_est(score.simplify_eq(best_solution[0]), 0, supervision_data, eta=self.eta)[0]
            if test_score >= 1 - self.norm_threshold:
                discovery_time = end_time
                break

        print('\n{} tests complete after {} iterations.'.format(i_test + 1, i_itr + 1))

        return best_solution, discovery_time, test_score
