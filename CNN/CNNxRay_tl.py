import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms, datasets, models, utils
import time
import numpy as np
from torchsummary import summary
from torch.utils.data import DataLoader

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def get_model():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False


    model.avgpool = AdaptiveConcatPool2d()
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1))
    return model


def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0


    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    writer.add_scalar('Train Loss', total_loss/len(train_loader), epoch)
    writer.flush()
    return total_loss/len(train_loader)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def misclassified_images(pred, writer, target, data, output, epoch, count=10):
    misclassified = (pred != target.data)
    for index, image_tensor in enumerate(data[misclassified][:count]):
        img_name = '{}->Predict-{}x{}-Actual'.format(
            epoch,
            LABEL[pred[misclassified].tolist()[index]],
            LABEL[target.data[misclassified].tolist()[index]],)
        writer.add_image(img_name, inv_normalize(image_tensor), epoch)


def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            misclassified_images(pred, writer, target, data, output, epoch)
    total_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('Test Loss', total_loss, epoch)
    writer.add_scalar('Accuracy', accuracy, epoch)
    writer.flush()
    return total_loss, accuracy


image_transforms = {
    'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=300, scale=(0.8, 1.1)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=256), # Image net standards
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Imagenet standards
]),
    'val':
        transforms.Compose([
            transforms.Resize(size=300),
            transforms.CenterCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    'test':
        transforms.Compose([
            transforms.Resize(size=300),
            transforms.CenterCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

datadir = "chest_xray/"
traindir = datadir + 'train/'
validdir = datadir + 'test/'
testdir = datadir + 'val/'
model_path = "model.pth"
batch_size = 128
PATH_to_log_dir = 'logdir/'


data = {
    'train':
        datasets.ImageFolder(root=traindir,
                             transform=image_transforms['train']),
    'val':
        datasets.ImageFolder(root=validdir,
                             transform=image_transforms['val']),
    'test':
        datasets.ImageFolder(root=testdir,
                             transform=image_transforms['test'])
}


dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size,
                        shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size,
                      shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size,
                       shuffle=True)
}

LABEL = dict((v,k) for k,v in data['train'].class_to_idx.items())