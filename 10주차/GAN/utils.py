import torch


def generate_real_samples(n):
    X1 = torch.rand(n) - 0.5
    X2 = X1 * X1

    X1 = X1.view(n, 1)
    X2 = X2.view(n, 1)

    X = torch.cat((X1, X2), dim=1)
    y = torch.ones(n, 1)
    return X, y


def generate_latent_points(latent_dim, n):
    x_input = torch.randn(n * latent_dim)
    x_input = x_input.view(n, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator(x_input)
    y = torch.zeros(n, 1)
    return X, y
