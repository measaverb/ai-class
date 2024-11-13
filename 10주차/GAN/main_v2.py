import matplotlib.pyplot as plt
import torch
from networks import DiscriminatorV2, GeneratorV2
from utils import generate_fake_samples, generate_real_samples


def train(
    generator,
    discriminator,
    criterion,
    optimizer_g,
    optimizer_d,
    latent_dim,
    n_batch=128,
    n_iter=100,
):
    for _ in range(n_iter):
        X_real, y_real = generate_real_samples(n_batch)
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_batch)

        discriminator.zero_grad()
        d_real = discriminator(X_real)
        d_fake = discriminator(X_fake)
        d_loss = criterion(d_real, y_real) + criterion(d_fake, y_fake)
        d_loss.backward()
        optimizer_d.step()

        generator.zero_grad()
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_batch)
        d_fake = discriminator(X_fake)
        g_loss = criterion(d_fake, y_real)
        g_loss.backward()
        optimizer_g.step()

    return d_loss, g_loss


def sample(epoch, generator, latent_dim, n=100):
    plt.figure()
    X_real, _ = generate_real_samples(n)
    X_real = X_real.detach().numpy()

    X_fake, _ = generate_fake_samples(generator, latent_dim, n)
    X_fake = X_fake.detach().numpy()

    plt.scatter(X_real[:, 0], X_real[:, 1], color="orange")
    plt.scatter(X_fake[:, 0], X_fake[:, 1], color="blue")

    plt.savefig(f"sampled_v2/epoch_{epoch}_sampled.png")


def main(hps):
    generator = GeneratorV2(hps["latent_dim"])
    discriminator = DiscriminatorV2()

    criterion = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=hps["lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=hps["lr"])

    for epoch in range(hps["epochs"]):
        if epoch % 100 == 0:
            sample(epoch, generator, hps["latent_dim"])
        d_loss, g_loss = train(
            generator=generator,
            discriminator=discriminator,
            criterion=criterion,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            latent_dim=hps["latent_dim"],
            n_batch=hps["batch_size"],
            n_iter=hps["n_iter"],
        )
        print(f"Epoch {epoch+1}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")


if __name__ == "__main__":
    hps = {
        "batch_size": 128,
        "n_iter": 100,
        "dataset_size": 100,
        "epochs": 1001,
        "lr": 1e-3,
        "latent_dim": 5,
    }
    main(hps)
