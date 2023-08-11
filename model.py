import time
from collections import deque

import numpy as np

import score
import symbolics
from MCTSblock import MCTSBlock
from policy_value_net import PolicyValueNetContext


class Model:
    def __init__(self, args):

        # Extract the properties from args
        properties = [
            'symbolic_lib',
            'max_len',
            'max_module_init',
            'num_transplant',
            'num_runs',
            'eta',
            'num_aug',
            'exploration_rate',
            'transplant_step',
            'norm_threshold',
            'device'
        ]

        for prop in properties:
            if hasattr(args, prop):
                setattr(self, prop, getattr(args, prop))
            else:
                raise ValueError(f'args does not have property {prop}')

        # Other initializations
        self.base_grammar = symbolics.rule_map[self.symbolic_lib]
        self.p_v_net_ctx = PolicyValueNetContext(self.base_grammar, self.num_transplant, self.device)

        self.nt_nodes = symbolics.ntn_map[self.symbolic_lib]
        self.score_with_est = score.score_with_est
        self.data_buffer = deque(maxlen=1024)

    def run(self, X, y=None):
        assert X.size(0) == 1
        if y is not None:
            X = X.squeeze(0)
            y = y.squeeze(0)

            time_idx = np.arange(X.size(0) + y.shape[0])
            input_data = np.vstack([time_idx[:X.size(0)], X])

            supervision_data = np.vstack([time_idx, np.concatenate([X, y])])
        else:
            X = X.squeeze(0)
            time_idx = np.arange(X.size(0))
            input_data = np.vstack([time_idx[:X.size(0)], X])
            supervision_data = np.vstack([time_idx, X])

        all_times = []
        all_eqs = []
        test_scores = []

        module_grow_step = (self.max_len - self.max_module_init) / self.num_transplant

        for i_test in range(self.num_runs):
            best_solution = ('nothing', 0)

            exploration_rate = self.exploration_rate  # 设置探索率
            max_module = self.max_module_init  # 设置最大模块
            reward_his = []  # 初始化奖励历史
            best_modules = []  # 初始化最佳模块
            aug_grammars = []  # 初始化增强语法

            start_time = time.time()  # 记录开始时间ß

            self.p_v_net_ctx.reset_grammar_vocab_name()

            for i_itr in range(self.num_transplant):
                mcts_block = MCTSBlock(data_sample=supervision_data,
                                       base_grammars=self.base_grammar,
                                       aug_grammars=aug_grammars,
                                       nt_nodes=self.nt_nodes,
                                       max_len=self.max_len,
                                       max_module=max_module,
                                       aug_grammars_allowed=self.num_aug,
                                       func_score=self.score_with_est,
                                       exploration_rate=self.exploration_rate,
                                       eta=self.eta)

                _, current_solution, good_modules, records = mcts_block.run(input_data,
                                                                            self.transplant_step,
                                                                            policy_value_net=self.p_v_net_ctx,
                                                                            num_play=10,
                                                                            print_flag=True)

                self.data_buffer.extend(list(records)[:])

                # 如果没有最佳模块，则将好的模块赋值给最佳模块
                if not best_modules:
                    best_modules = good_modules
                else:
                    # 否则，将最佳模块和好的模块合并，并按照评分进行排序
                    best_modules = sorted(list(set(best_modules + good_modules)), key=lambda x: x[1])

                # 更新增强语法
                aug_grammars = [x[0] for x in best_modules[-self.num_aug:]]

                # 将最佳解决方案的评分添加到奖励历史中
                reward_his.append(best_solution[1])

                # 如果当前解决方案的评分大于最佳解决方案的评分，则更新最佳解决方案
                if current_solution[1] > best_solution[1]:
                    best_solution = current_solution

                # 增加最大模块
                max_module += module_grow_step
                # 增加探索率
                exploration_rate *= 5

                # 检查是否发现了解决方案。如果是，提前停止。
                test_score = \
                    self.score_with_est(score.simplify_eq(best_solution[0]), 0, supervision_data, eta=self.eta)[0]
                # if test_score >= 1 - self.norm_threshold:
                #     num_success += 1
                #     if discovery_time == 0:
                #         discovery_time = end_time
                #         all_times.append(discovery_time)
                #     break

            all_eqs.append(score.simplify_eq(best_solution[0]))
            test_scores.append(test_score)

            print('\n{} tests complete after {} iterations.'.format(i_test + 1, i_itr + 1))
            print('best solution: {}'.format(score.simplify_eq(best_solution[0])))
            print('test score: {}'.format(test_score))
            print()

        # 计算成功率
        # success_rate = num_success / self.num_runs
        # if count_success:
        #     print('success rate :', success_rate)

        # 返回所有发现的方程、成功率和运行时间
        return all_eqs, all_times, test_scores, supervision_data
