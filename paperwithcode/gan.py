import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    img = (img+1)/2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap="gray")
    plt.show()


def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img+1)/2

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

standardizator = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5),   # 3 for RGB channels이나 실제론 gray scale
                         std=(0.5))])  # 3 for RGB channels이나 실제론 gray scale


train_data = dsets.MNIST(root="data/", train=True,
                         transform=standardizator, download=True)
test_data = dsets.MNIST(root="data/", train=False,
                        transform=standardizator, download=True)

batch_size = 200
train_data_loader = DataLoader(train_data, batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size, shuffle=True)


# Hyper Parameters
d_noise = 100
d_hidden = 256
k = 4


def sample_z(batch_size=1, d_noise=100):
    return torch.randn(batch_size, d_noise, device=device)


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_noise, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 28*28),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layer(x)


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(28*28, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


criterion = nn.BCELoss()


def run_epoch(generator, discriminator, _optimizer_g, _optimizer_d):
    generator.train()
    discriminator.train()

    for epoch, (img_batch, label_batch) in enumerate(train_data_loader):
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        if (epoch+1) % 5 != 0:
            # maximize V(discriminator,generator) = optimize discriminator (setting k to be 1)  #
            _optimizer_d.zero_grad()

            p_real = discriminator(img_batch.view(-1, 28*28))
            p_fake = discriminator(generator(sample_z(batch_size, d_noise)))

            loss_real = -1*torch.log(p_real)
            loss_fake = -1*torch.log(1.-p_fake)
            loss_d = (loss_real+loss_fake).mean()

            loss_d.backward()
            _optimizer_d.step()
        else:
            # minimize V(discriminator, generator) #
            _optimizer_g.zero_grad()

            p_fake = discriminator(generator(sample_z(batch_size, d_noise)))

            loss_g = -1*torch.log(p_fake).mean()

            loss_g.backward()
            _optimizer_g.step()


def evaluate_model(generator, discriminator):
    p_real, p_fake = 0., 0.

    generator.eval()
    discriminator.eval()

    for img_batch, label_batch in test_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device
                                                                      )

        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28*28))).item()) / \
                10000.

            p_fake += (torch.sum(discriminator(generator(sample_z(batch_size, d_noise)))).item()) / \
                10000.

    return p_real, p_fake


def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)


if __name__ == "__main__":
    G_model = G().to(device)
    D_model = D().to(device)

    init_params(G_model)
    init_params(D_model)

    # z = sample_z()
    # # img_fake = G_model(z).view(-1, 28, 28)
    # # imshow(img_fake.squeeze().cpu().detach())

    # # z = sample_z(batch_size)
    # # img_fake = G_model(z)
    # # imshow_grid(img_fake)

    # print(G_model(z).shape)
    # print(D_model(G_model(z)).shape)

    optimizer_g = optim.Adam(G_model.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(D_model.parameters(), lr=0.0002)

    p_real_trace = []
    p_fake_trace = []

    for epoch in range(200):
        run_epoch(G_model, D_model, optimizer_g, optimizer_d)
        p_real, p_fake = evaluate_model(G_model, D_model)

        p_real_trace.append(p_real)
        p_fake_trace.append(p_fake)

        if ((epoch+1) % 50 == 0):
            print(f"[Epoch {epoch}/200] p_real: {p_real:3} p_g {p_fake:3}")
            imshow_grid(G_model(sample_z(16)).view(-1, 1, 28, 28))

    plt.plot(p_fake_trace, label='D(x_generated)')
    plt.plot(p_real_trace, label='D(x_real)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

    torch.save(G_model.state_dict(), "gan.pt")
