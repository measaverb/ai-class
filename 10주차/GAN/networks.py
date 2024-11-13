import torch.nn as nn


class GeneratorV1(nn.Module):
    def __init__(self, latent_dim, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15, out_dim)

    def _initialise_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DiscriminatorV1(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def _initialise_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class GeneratorV2(nn.Module):
    def __init__(self, latent_dim, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 25)
        self.fc2 = nn.Linear(25, 15)
        self.fc3 = nn.Linear(15, out_dim)
        self.lrelu = nn.LeakyReLU()

    def _initialise_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.fc3(x)
        return x


class DiscriminatorV2(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 25)
        self.fc2 = nn.Linear(25, 15)
        self.fc3 = nn.Linear(15, 5)
        self.fc4 = nn.Linear(5, 1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def _initialise_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.fc3(x)
        x = self.lrelu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
