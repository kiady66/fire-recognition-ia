import os
from skimage import io, transform
import numpy as np
from tqdm import tqdm
from model import FireNet
from data_augmentation import prepare_dataset

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch.utils.tensorboard import SummaryWriter

prepare_dataset()
writer = SummaryWriter('runs/firenet_experiment_1')
print(torch.cuda.is_available())
net = FireNet()
device = "cuda" if torch.cuda.is_available() else "cpu"
net.to(device)

TRAINING_PATH = 'training_dataset'
CATEGORIES = ['Fire', 'NoFire']


class TrainingSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        classes = []
        images = []
        for category in CATEGORIES:
            path = os.path.join(TRAINING_PATH, category)
            class_num = CATEGORIES.index(category)

            for img in tqdm(os.listdir(path)):
                try:
                    image = io.imread(os.path.join(path, img))
                    if (image.shape[2] == 3):
                        if (self.transform is not None):
                            image = self.transform(image)
                            images.append(image)
                            classes.append(class_num)
                except Exception as e:
                    pass
        self.set = {'image': images, 'class': classes}

    def __len__(self):
        return len(self.set['class'])

    def __getitem__(self, idx):
        image = self.set['image'][idx]
        classe = self.set['class'][idx]
        sample = {'image': image, 'class': classe}
        return sample


training_set = TrainingSet(transform=transforms.ToTensor())
print(len(training_set))


class TestingSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        classes = []
        images = []
        for category in CATEGORIES:
            path = os.path.join('testing_dataset', category)
            class_num = CATEGORIES.index(category)

            for img in tqdm(os.listdir(path)):
                try:
                    image = io.imread(os.path.join(path, img))
                    if (image.shape[2] == 3):
                        if (self.transform):
                            image = self.transform(image)
                            images.append(image)
                            classes.append(class_num)
                except Exception as e:
                    pass
        self.set = {'image': images, 'class': classes}

    def __len__(self):
        return len(self.set['class'])

    def __getitem__(self, idx):
        image = self.set['image'][idx]
        classe = self.set['class'][idx]
        sample = {'image': image, 'class': classe}
        return sample


test_set = TestingSet(transform=transforms.ToTensor())
print(len(test_set))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), eps=1e-7)

trainloader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
net = net.float()
net.train()

len(trainloader)

for epoch in range(100):
    global_loss = 0.0
    for data in tqdm(trainloader):
        inputs, labels = data['image'].to(device), data['class'].to(device)

        optimizer.zero_grad()

        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        global_loss += loss.item()
    print('global loss for epoch %d : %.3f' % (epoch + 1, global_loss / len(trainloader)))
    writer.add_scalar('Training loss 100', global_loss / len(trainloader), epoch + 1)

torch.save(net.state_dict(), './trained_weights100.pth')
net = FireNet()
net.float()
net.cuda()
net.load_state_dict(torch.load('./trained_weights100.pth'))
net.eval()

trainloader = DataLoader(training_set, batch_size=4, shuffle=False, num_workers=0)
net = net.float()
net.eval()

correct = 0
total = 0
with torch.no_grad():
    print('evaluate accuracy on training set:')
    for data in tqdm(trainloader):
        images, labels = data['image'].to(device), data['class'].to(device)
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
print('accuracy: %d' % (100 * correct / total))

testloader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0)
net = net.float()
net.eval()


len(testloader)

correct = 0
total = 0
with torch.no_grad():
    print('evaluate accuracy on training set:')
    for data in tqdm(testloader):
        images, labels = data['image'].to(device), data['class'].to(device)
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
print ('accuracy: %d' % (100*correct/total))
