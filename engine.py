import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as op
from scipy.stats import pearsonr
from sympy import symbols, lambdify, sympify
from torch.distributions import Categorical

from model import Model


class Engine(object):
    def __init__(self, args):
        self.args = args
        self.model = Model(args)
        self.optimizer = op.Adam(self.model.p_v_net_ctx.pv_net.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    @staticmethod
    def metrics(exps, scores, data):
        best_index = np.argmax(scores)
        best_exp = exps[best_index]
        span, gt = data
        x = symbols("x")
        expression = sympify(best_exp)
        f = lambdify(x, expression, "numpy")
        prediction = f(span)
        mae = np.mean(np.abs(prediction - gt))
        mse = np.mean((prediction - gt) ** 2)
        corr, _ = pearsonr(prediction, gt)

        return mae, mse, corr

    def simulate(self, inputs):
        X, y = inputs[:, :self.args.seq_in], inputs[:, -self.args.seq_out:]

        all_eqs, all_times, test_scores, test_data = self.model.run(X, y)
        mae, mse, corr = self.metrics(all_eqs, test_scores, test_data)
        if len(self.model.data_buffer) > self.args.train_size:
            loss = self.train()
            return all_eqs, all_times, test_data, loss.item(), mae, mse, corr
        return all_eqs, all_times, test_data, 0, mae, mse, corr

    def train(self):
        print("start train neural networks...")
        self.optimizer.zero_grad()
        self.model.p_v_net_ctx.pv_net.train()
        state_batch, seq_batch, policy_batch, value_batch, length_indices = self.preprecess_data()
        value_batch = torch.Tensor(value_batch)
        raw_dis_out, value_out = self.model.p_v_net_ctx.policy_value_batch(seq_batch, state_batch)
        total_loss = F.mse_loss(value_out, value_batch)
        for length, sample_id in length_indices.items():
            gt_policy = torch.Tensor([policy_batch[i] for i in sample_id])
            out_policy = F.softmax(torch.stack([raw_dis_out[i] for i in sample_id])[:, :length], dim=-1)
            dist_target = Categorical(probs=gt_policy)
            dist_out = Categorical(probs=out_policy)
            total_loss += torch.distributions.kl_divergence(dist_target, dist_out).mean()
        total_loss.backward()
        if self.args.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.p_v_net_ctx.pv_net.parameters(), self.args.clip)
        self.optimizer.step()
        print("end train neural networks...")
        return total_loss

    def obtain_policy_length(self, policy):
        length_indices = defaultdict(list)
        for idx, sublist in enumerate(policy):
            length_indices[len(sublist)].append(idx)
        return dict(length_indices)

    def preprecess_data(self):
        mini_batch = random.sample(self.model.data_buffer, self.args.train_size)
        state_batch = [data[0] for data in mini_batch]
        seq_batch = [data[1][1] for data in mini_batch]
        policy_batch = [data[2] for data in mini_batch]
        value_batch = [data[3] for data in mini_batch]
        length_indices = self.obtain_policy_length(policy_batch)
        return state_batch, seq_batch, policy_batch, value_batch, length_indices

    def eval(self, inputs):
        self.model.p_v_net_ctx.pv_net.eval()
        X, y = inputs[:, :self.args.seq_in], inputs[:, -self.args.seq_out:]

        all_eqs, all_times, test_data = self.model.run(X, y)
