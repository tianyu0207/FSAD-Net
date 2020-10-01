import torch
from models import *
from torch.utils.data import DataLoader
from Recorder import Recorder
from tqdm import tqdm
from pathlib import Path
import torch.nn.init as init
import os
import torchvision
from complex_2d_my_data_loader import MyDataLoader
import numpy
from sampler import BalancedBatchSampler

recorder = Recorder('colon', 'Colonoscopy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-4
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
mone = mone.to(device)

def weights_init(m):

    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


class Contrastive_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        confidence_margin = torch.tensor(6, dtype=torch.float).to(device)
        zero = torch.tensor(0., dtype=torch.float).to(device)
        y_true = y_true.to(device)
        ref = torch.tensor(numpy.random.normal(loc=0., scale=1.0, size=15000), dtype=torch.float)
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.max(confidence_margin - dev, zero))

        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss), torch.mean((1 - y_true) * inlier_loss), torch.mean(y_true * inlier_loss)


data_root = '/home/yu/PycharmProjects/MICCAI/data/colon'


data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.CenterCrop(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

epoch = 5660

encoder = Encoder().to(device)

root = Path('/home/yu/PycharmProjects/MICCAI2020/check_points/encoder') #deep infomax pretrained encoder
enc_file = root / Path('encoder_5660.wgt')
encoder.load_state_dict(torch.load(enc_file))

encoder.eval()

con_criterion = Contrastive_Loss().to(device)
dev_net = dev_network_d().to(device)
dev_net.apply(weights_init)
dev_net.train()
optimizer_dev = torch.optim.Adam(dev_net.parameters(), lr=LR, betas=(0, 0.9))
d_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dev, step_size=10, gamma=0.95)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    data_root = '/home/yu/PycharmProjects/MICCAI2020/data/colon/train'
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    training_dataset = MyDataLoader(normal_path=os.path.join(data_root, 'train_set'),
                                    transform=data_transform,
                                    abnormal_path=os.path.join(data_root, 'abnormal_50'),
                                    test_path=None,
                                    train=True)

    count = 0
    labels = []
    for index in range(0, len(training_dataset)):
        _, label = training_dataset.__getitem__(index)
        if label.split('/')[-1].split('_')[0] == '0':

            count += 1
            # print(label)
            labels.append(1)
            # abnormal sample
        else:
            labels.append(0)
            # normal sample
    print("The number of abnormal in training set: " + str(count))
    labels = numpy.array(labels)
    target = torch.tensor(labels).to(device)

    training_dataset_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False,
                                                          drop_last=True, sampler= BalancedBatchSampler(training_dataset, target),
                                                          num_workers=0, pin_memory=True)

    epoch_restart = 0
    dataiter = iter(training_dataset_loader)

    for epoch in range(epoch_restart + 1, 1101):
        batch = tqdm(training_dataset_loader, total=len(training_dataset) // batch_size)
        new_count = 0
        for x, label in batch:
            x = x.to(device)
            l = []
            for index in range(0, x.shape[0]):

                if label[index].split('/')[-1].split('_')[0] == '0':
                    count += 1
                    l += [1]
                else:
                    l += [0]

            l = numpy.array(l)
            l = torch.tensor(l, dtype=torch.float).to(device)

            target = []
            for index in range(0, x.shape[0]):

                if label[index].split('/')[-1].split('_')[0] == '0':
                    count += 1
                    target += [1]
                    # abnormal sample
                else:
                    target += [-1]
                    # normal sample

            target = numpy.array(target)
            target = torch.tensor(target, dtype=torch.float).to(device)

            l_reverse = []
            for index in range(0, x.shape[0]):

                if label[index].split('/')[-1].split('_')[0] == '0':
                    count += 1
                    # print(label)
                    l_reverse += [0]
                    # abnormal sample
                else:
                    l_reverse += [1]
                    # normal sample
            l_reverse = numpy.array(l_reverse)
            l_reverse = torch.tensor(l_reverse, dtype=torch.float).to(device)

            encodings, features_in = encoder(x)
            optimizer_dev.zero_grad()

            scores = dev_net(encodings)
            scores = torch.squeeze(scores)

            outlier_scores = l_reverse * scores
            inlier_scores = l * scores
            Contra_Loss, inlier_loss, outlier_loss = con_criterion(l, scores)

            Contra_Loss.backward()
            optimizer_dev.step()
            recorder.record(loss=Contra_Loss, epoch=int(epoch), num_batches=len(training_dataset_loader), n_batch=epoch,
                            loss_name='dev contrastive loss')
            recorder.record(loss=outlier_scores.mean(), epoch=int(epoch), num_batches=len(training_dataset_loader), n_batch=epoch,
                            loss_name='inlier score')
            recorder.record(loss=inlier_scores.mean(), epoch=int(epoch), num_batches=len(training_dataset_loader), n_batch=epoch,
                            loss_name='outlier score')
        if epoch % 10 == 0:
            path = 'check_points/'
            torch.save(dev_net.state_dict(), path + '/ckpt/fsad_net_{}.pth'.format(str(epoch)))

