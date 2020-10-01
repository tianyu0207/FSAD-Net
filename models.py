from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn


class Encoder(nn.Module):
    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
        self.l1 = nn.Linear(512*20*20, self.rep_dim)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))
        # print(encoded.shape)
        return encoded, features

    def get_embedding(self, x):
        return self.forward(x)

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 22 * 22 + 64, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz


class DeepInfoAsLatent(nn.Module):
    def __init__(self, run, epoch):
        super().__init__()
        model_path = Path('/home/yu/PycharmProjects/DeepInfomaxPytorch-master/data/cifar/checkpoints') / Path(str(run)) / Path('encoder' + str(epoch) + '.wgt')
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(str(model_path)))
        self.classifier = Classifier()

    def forward(self, x):
        z, features = self.encoder(x)
        z = z.detach()
        return self.classifier((z, features))

DIM = 64

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()


        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
        self.l1 = nn.Linear(512 * 20 * 20, 64)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)

        preprocess = nn.Sequential(
            nn.Linear(64, 20*20*512),
            nn.BatchNorm1d(20*20*512),

            nn.ReLU(inplace=True),

        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(64, 3, 4, stride=1)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.block2 = block3
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))


        output = self.preprocess(encoded)

        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)

        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output1  = output
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)

        return output, output1

class dev_network_d(nn.Module):
    def __init__(self):
        super(dev_network_d, self).__init__()
        self.l0 = nn.Linear(64, 1000)
        self.n0 = nn.BatchNorm1d(num_features=1000)
        self.l1 = nn.Linear(1000, 200)
        self.n1 = nn.BatchNorm1d(num_features=200)
        self.l2 = nn.Linear(200, 20)
        self.n2 = nn.BatchNorm1d(num_features=20)
        self.l3 = nn.Linear(20, 1)

    def forward(self, input):
        h = F.relu(self.l0(input))
        h = self.n0(h)
        h = F.relu(self.l1(h))
        h = self.n1(h)
        h = F.relu( self.l2(h))
        h = self.n2(h)
        h = self.l3(h)

        return h



class classifer_d(nn.Module):
    def __init__(self):
        super(classifer_d, self).__init__()
        self.l0 = nn.Linear(64, 1000)
        self.n0 = nn.BatchNorm1d(num_features=1000)
        self.l1 = nn.Linear(1000, 200)
        self.n1 = nn.BatchNorm1d(num_features=200)
        self.l2 = nn.Linear(200, 20)
        self.n2 = nn.BatchNorm1d(num_features=20)
        self.l3 = nn.Linear(20, 2)
        self.n3 = nn.BatchNorm1d(num_features=2)

    def forward(self, input):

        h = F.relu(self.l0(input))
        h = self.n0(h)
        h = F.relu(self.l1(h))
        h = self.n1(h)
        h = F.relu( self.l2(h))
        h = self.n2(h)
        h = self.l3(h)
        h = self.n3(h)
        # h = F.softmax(h, dim=1)

        return h