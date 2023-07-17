import _thread
import threading
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from scipy.optimize import minimize

from symbolics import rule_map, ntn_map


class Context(object):
    def __init__(self, lib):
        self.lib = rule_map[lib]
        self.ntn = ntn_map[lib]
        self.QN = defaultdict(lambda: np.zeros(2))
        self.UCBs = defaultdict(lambda: np.zeros(len(self.lib)))

    def valid_prods(self, Node):
        """
        获取所有以给定节点开始的可能的产生规则的索引。
        """
        # 通过检查每个语法规则是否以给定的节点开始，找出所有有效的语法规则。
        valid_grammars = [x for x in self.lib if x.startswith(Node)]
        # 返回有效语法规则在总语法规则列表中的索引。
        return [self.lib.index(x) for x in valid_grammars]

    def get_policy1(self, nA):
        """
        Creates a policy based on UCB score.
        """

        def policy_fn(state, node):
            valid_actions = self.valid_prods(node)
            ucb_scores = self.UCBs[state][valid_actions]
            ucb_scores /= sum(ucb_scores)

            A = np.zeros(nA, dtype=float)
            best_action = valid_actions[np.argmax(ucb_scores)]
            A[best_action] += 0.8
            A[valid_actions] += float(0.2 / len(valid_actions))

            return A

        return policy_fn

    def get_policy2(self, nA):
        """
        Creates an random policy to select an unvisited child.（均匀分布）
        """

        def policy_fn(UC):
            if len(UC) != len(set(UC)):
                print(UC)
                print(self.lib)
            A = np.zeros(nA, dtype=float)
            A[UC] += float(1 / len(UC))
            return A

        return policy_fn

    def available_action(self, node, state):
        all_valid_actions = [x for x in self.lib if x.startswith(node)]
        all_valid_action_idx = [self.lib.index(x) for x in all_valid_actions]
        valid_actions_unvisited = [a for a in all_valid_action_idx if
                                   self.QN[state + ',' + self.lib[a]][1] == 0]
        return valid_actions_unvisited


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


class MCTSUtility:

    @staticmethod
    def tree_to_eq(prods):
        """
        将解析树转换为等式形式。
        """
        seq = ['f']
        for prod in prods:
            if str(prod[0]) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod[0]:
                    seq1 = seq[:ix]
                    seq2 = list(prod[3:])
                    seq3 = seq[ix + 1:]
                    seq = seq1 + seq2 + seq3
                    break
        try:
            return ''.join(seq)
        except:
            return ''

    @staticmethod
    def score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999):
        ## 定义独立变量和因变量
        num_var = data.shape[0] - 1
        if num_var <= 3:  ## 大部分情况 ([x], [x,y], 或 [x,y,z])
            current_var = 'x'
            for i in range(num_var):
                globals()[current_var] = data[i, :]
                current_var = chr(ord(current_var) + 1)
            globals()['f_true'] = data[-1, :]
        else:  ## 目前只有双摆案例有超过3个独立变量
            globals()['x1'] = data[0, :]
            globals()['x2'] = data[1, :]
            globals()['w1'] = data[2, :]
            globals()['w2'] = data[3, :]
            globals()['wdot'] = data[4, :]
            globals()['f_true'] = data[5, :]

        ## 计算eq中数值的数量
        c_count = eq.count('C')
        with time_limit(t_limit, 'sleep'):
            try:
                if c_count == 0:  ## 无数值
                    f_pred = eval(eq)
                elif c_count >= 10:  ## 不鼓励过于复杂的数值估计
                    return 0, eq
                else:  ## 有数值：使用Powell方法进行系数估计

                    c_lst = ['c' + str(i) for i in range(c_count)]
                    for c in c_lst:
                        eq = eq.replace('C', c, 1)

                    def eq_test(c):
                        for i in range(len(c)):
                            globals()['c' + str(i)] = c[i]
                        return np.linalg.norm(eval(eq) - f_true, 2)

                    x0 = [1.0] * len(c_lst)
                    c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist()
                    c_lst = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in c_lst]
                    eq_est = eq
                    for i in range(len(c_lst)):
                        eq_est = eq_est.replace('c' + str(i), str(c_lst[i]), 1)
                    eq = eq_est.replace('+-', '-')
                    f_pred = eval(eq)
            except:
                return 0, eq

        r = float(eta ** tree_size / (1.0 + np.linalg.norm(f_pred - f_true, 2) ** 2 / f_true.shape[0]))

        return r, eq


class MCTSPlayer(object):
    def __init__(self, max_aug, max_module, exploration_rate, scale, max_len, lib_name, num_episodes, num_play):
        self.context = Context(lib=lib_name)
        self.states = []
        self.action_num = len(self.context.lib)
        self.policy1 = self.context.get_policy1(self.action_num)
        self.policy2 = self.context.get_policy2(self.action_num)
        self.best_solution = ('nothing', 0)
        self.num_episodes = num_episodes
        self.max_len = max_len
        self.reward_his = []
        self.good_modules = []
        self.max_module = max_module
        self.scale = scale
        self.exploration_rate = exploration_rate
        self.max_aug = max_aug
        self.num_play = num_play

    def run(self, X):
        for i_episode in range(1, self.num_episodes + 1):
            state = 'f->A'
            ntn = ['A']
            unvisited_children = self.context.available_action(ntn[0], state)
            while not unvisited_children:
                action = np.random.choice(np.arange(self.action_num), p=self.policy1(state, ntn[0]))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn, X)
                if state not in self.states:
                    self.states.append(state)

                if not done:
                    state = next_state
                    ntn = ntn_next
                    unvisited_children = self.context.available_action(ntn[0], state)
                    if state.count(',') >= self.max_len:
                        unvisited_children = []
                        self.backpropogate(state, action, 0)
                        self.reward_his.append(self.best_solution[1])
                        break
                else:
                    unvisited_children = []
                    if reward > self.best_solution[1]:
                        self.best_solution = (eq, reward)
                        self.update_modules(state, reward, eq)
                        self.update_QN_scale(reward)

                    self.backpropogate(state, action, reward)
                    self.reward_his.append(self.best_solution[1])
                    break

            if unvisited_children:
                action = np.random.choice(np.arange(self.action_num), p=self.policy2(unvisited_children))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn, X)

                if not done:
                    reward, eq = self.rollout(self.num_play, next_state, ntn_next, X)
                    if state not in self.states:
                        self.states.append(state)

                if reward > self.best_solution[1]:
                    self.update_QN_scale(reward)
                    self.best_solution = (eq, reward)

                self.backpropogate(state, action, reward)
                self.reward_his.append(self.best_solution[1])

        return self.reward_his, self.best_solution, self.good_modules

    def rollout(self, num_play, state_initial, ntn_initial, X):
        best_eq = ''
        best_r = 0

        for n in range(num_play):
            # 初始化done为False，表示模拟未完成。
            # state和ntn被初始化为传入的初始值。
            done = False
            state = state_initial
            ntn = ntn_initial

            # 当模拟未完成时，执行以下操作。
            while not done:
                # 调用valid_prods函数计算ntn的有效索引。
                valid_index = self.valid_prods(ntn[0])

                # 从有效索引中随机选择一个作为行动。
                action = np.random.choice(valid_index)

                # 调用step函数来执行行动，并获取下一状态、下一ntn、奖励、是否完成以及方程。
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn, X)

                # 更新状态和ntn为下一状态和下一ntn。
                state = next_state
                ntn = ntn_next

                # 如果状态中的逗号数量大于或等于最大长度，中断循环。
                if state.count(',') >= self.max_len:
                    break

            # 如果模拟完成，执行以下操作。
            if done:
                # 如果奖励大于最高奖励，更新最高奖励和最好的方程，并使用update_modules函数更新模块。
                if reward > best_r:
                    self.update_modules(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward

        # 函数返回最高奖励和最好的方程。
        return best_r, best_eq

    def valid_prods(self, Node):
        """
        获取所有以给定节点开始的可能的产生规则的索引。
        """
        # 通过检查每个语法规则是否以给定的节点开始，找出所有有效的语法规则。
        valid_grammars = [x for x in self.context.lib if x.startswith(Node)]
        # 返回有效语法规则在总语法规则列表中的索引。
        return [self.context.lib.index(x) for x in valid_grammars]

    def update_QN_scale(self, new_scale):
        """
        Update the Q values self.scaled by the new best reward.
        此方法更新Q值。
        """

        if self.scale != 0:
            for s in self.context.QN:
                self.context.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def update_modules(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an
        single action for future use.
        如果我们经过一个具有高分的简洁解决方案，我们将其存储为以后使用的单个动作。
        """
        module = state[5:]
        if state.count(',') <= self.max_module:
            if not self.good_modules:
                self.good_modules = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.good_modules]:
                if len(self.good_modules) < self.max_aug:
                    self.good_modules = sorted(self.good_modules + [(module, reward, eq)], key=lambda x: x[1])
                else:
                    if reward > self.good_modules[0][1]:
                        self.good_modules = sorted(self.good_modules[1:] + [(module, reward, eq)], key=lambda x: x[1])

    def backpropogate(self, state, action_index, reward):
        action = self.context.lib[action_index]
        if self.scale != 0:
            self.context.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.context.QN[state + ',' + action][0] += 0

        self.context.QN[state + ',' + action][1] += 1

        while state:
            if self.scale != 0:
                self.context.QN[state][0] += reward / self.scale
            else:
                self.context.QN[state][0] += 0

            self.context.QN[state][1] += 1
            self.context.UCBs[state][self.context.lib.index(action)] = self.update_ucb_mcts(state, action)

            if ',' in state:
                state, action = state.rsplit(',', 1)
            else:
                state = ''

    def update_ucb_mcts(self, state, action):
        next_state = state + ',' + action
        Q_child = self.context.QN[next_state][0]
        N_parent = self.context.QN[state][1]
        N_child = self.context.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)

    def get_ntn(self, prod, prod_idx):
        if prod_idx >= len(self.context.lib):
            return []
        else:
            ret = [i for i in prod[3:] if i in self.context.ntn]
            return ret

    def step(self, state, action_idx, ntn, X):
        action = self.context.lib[action_idx]
        state = state + ',' + action
        ntn = self.get_ntn(action, action_idx) + ntn[1:]
        if not ntn:
            reward, eq = MCTSUtility.score_with_est(MCTSUtility.tree_to_eq(state.split(',')),
                                                    len(state.split(',')), X)
            return state, ntn, reward, True, eq
        else:
            return state, ntn, 0, False, None
