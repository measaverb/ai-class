import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar):
    reconstruction_term = F.binary_cross_entropy(
        recon_x, x.view(-1, 28 * 28), reduction="sum"
    )
    regularisation_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_term + regularisation_term
