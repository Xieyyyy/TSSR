import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, grammar_vocab, hidden_dim=16):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.embedding_table = nn.Embedding(len(self.grammar_vocab), hidden_dim)

    def forward(self, seq, state_idx):
        state = self.embedding_table(state_idx)



class PolicyValueNetContext:
    """policy-value network context """

    def __init__(self, grammrs, num_transplant):
        self.grammar_vocab = ['f->A'] + grammrs
        self.num_transplant = num_transplant
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        self.pv_net = PVNet(self.grammar_vocab)

    def policy_value(self, seq, state):
        state_idx = torch.Tensor([self.symbol2idx[item] for item in state])
        policy = self.pv_net(seq, state_idx)

    def policy_value_fn(self):
        pass

    def train_step(self):
        pass
