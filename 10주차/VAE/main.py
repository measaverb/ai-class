from networks import VAE
from torchvision.datasets import MNIST
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import vae_loss
import matplotlib.pyplot as plt


def train(net, train_loader, optimizer, device):
    net.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(torch.float32)
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = net(data)

        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss


def sample(epoch, net, device, num_samples=10, latent_dim=20):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        z = z.to(device)
        sampled = net.decode(z).view(-1, 1, 28, 28)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(sampled[i][0].cpu(), cmap="gray")
        ax.axis("off")

    plt.savefig(f"sampled/epoch_{epoch}_sampled.png")
    plt.close(fig)


def main(hps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_data = MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=hps["batch_size"], shuffle=True)

    net = VAE(latent_dim=hps["latent_dim"])
    optimizer = optim.Adam(net.parameters(), lr=hps["lr"])

    for epoch in range(hps["epochs"]):
        train_loss = train(net, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")
        if epoch % 10 == 0 or epoch == 99:
            sample(epoch, net, device, num_samples=10, latent_dim=hps["latent_dim"])


if __name__ == "__main__":
    hps = {"batch_size": 128, "epochs": 100, "lr": 1e-3, "latent_dim": 20}
    main(hps)
