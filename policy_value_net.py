import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, grammar_vocab, hidden_dim=16):
        super(PVNet, self).__init__()
        self.grammar_vocab = grammar_vocab
        self.embedding_table = nn.Embedding(len(self.grammar_vocab), hidden_dim)
        self.lstm_state = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.lstm_seq = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)

    def forward(self, seq, state_idx):
        state = self.embedding_table(state_idx.long()).unsqueeze(0)
        seq = torch.Tensor(seq[1, :]).unsqueeze(0).unsqueeze(-1)

        out_state, _ = self.lstm_state(state)
        out_seq, _ = self.lstm_seq(seq)

        out = torch.cat([out_state[:, -1, :], out_seq[:, -1, :]], dim=-1)


class PolicyValueNetContext:
    """policy-value network context """

    def __init__(self, grammrs, num_transplant):
        self.grammar_vocab = ['f->A'] + grammrs
        self.num_transplant = num_transplant
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.grammar_vocab)}
        self.pv_net = PVNet(self.grammar_vocab)

    def policy_value(self, seq, states):
        if len(states) == 0:
            states.append('f->A')
        state_idx = torch.Tensor([self.symbol2idx[item] for item in states])
        policy = self.pv_net(seq, state_idx)

    def policy_value_fn(self):
        pass

    def train_step(self):
        pass
