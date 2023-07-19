import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lin = nn.Linear(10, 10)

    def forward(self, X, state):
        print(X.shape)
        print()
