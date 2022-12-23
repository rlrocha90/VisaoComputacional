import torch
from torch import nn
from torchvision import models
import numpy as np
import time
import torchvision.transforms as transforms
from torchvision import datasets
import os

path2data = "./data"

if not os.path.exists(path2data):
    os.mkdir(path2data)
data_transformer = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.STL10(path2data, split='train', download=True, transform=data_transformer) #5,000 imagens 3*96*96

import collections

# Contagem de número de imagens em cada categoria no train_ds
y_train = [y for _, y in train_ds]
counter_train = collections.Counter(y_train)

# Carregar os dados de teste em test0_ds
test0_ds = datasets.STL10(path2data, split='test', download=True, transform=data_transformer)

# Dividir os índices do conjunto test0_ds em dois grupos
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = list(range(len(test0_ds)))
y_test0 = [y for _, y in test0_ds]
for test_index, val_index in sss.split(indices, y_test0):
    print("test:", test_index, "val:", val_index)
    print(len(val_index), len(test_index))

# Criar dois dataset a partir de test0_ds
from torch.utils.data import Subset

val_ds = Subset(test0_ds, val_index)
test_ds = Subset(test0_ds, test_index)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CARREGAR O MODELO...
# Carregar o modelo Resnet18
model_resnet18 = models.resnet18(weights=None)
num_ftrs = model_resnet18.fc.in_features
num_classes = 10
model_resnet18.fc = nn.Linear(num_ftrs, num_classes)

# Carregar os dados armazenados do modelo treinado previamente

import torch

path2weights = "./models/resnet18_pretrained.pt"
model_resnet18.load_state_dict(torch.load(path2weights))

model_resnet18.eval()
if torch.cuda.is_available():
    device = torch.device("cuda")
    model_resnet18 = model_resnet18.to(device)


# Inicialização

def deploy_model(model, dataset, device, num_classes=10, sanity_check=False):
    len_data = len(dataset)
    y_out = torch.zeros(len_data, num_classes)
    y_gt = np.zeros((len_data), dtype="uint8")
    model = model.to(device)
    elapsed_times = []
    with torch.no_grad():
        for i in range(len_data):
            x, y = dataset[i]
            y_gt[i] = y
            start = time.time()
            yy = model(x.unsqueeze(0).to(device))
            y_out[i] = torch.softmax(yy, dim=1)
            elapsed = time.time() - start
            elapsed_times.append(elapsed)
            if sanity_check is True:
                break
    inference_time = np.mean(elapsed_times) * 1000
    print("average inference time per image on %s: %.2f ms "
          % (device, inference_time))
    return y_out.numpy(), y_gt


y_out, y_gt = deploy_model(model_resnet18, val_ds, device=device, sanity_check=False)
print(y_out.shape, y_gt.shape)

from sklearn.metrics import accuracy_score

y_pred = np.argmax(y_out, axis=1)
print(y_pred.shape, y_gt.shape)
acc = accuracy_score(y_pred, y_gt)
print("accuracy: %.2f" % acc)

# Implementar no test_ds
y_out, y_gt = deploy_model(model_resnet18, test_ds, device=device)
y_pred = np.argmax(y_out, axis=1)
acc = accuracy_score(y_pred, y_gt)
print(acc)

from torchvision import utils
import matplotlib.pyplot as plt

np.random.seed(1)


def imshow(inp, title=None):
    mean = [0.4467106, 0.43980986, 0.40664646]
    std = [0.22414584, 0.22148906, 0.22389975]
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


grid_size = 4
rnd_inds = np.random.randint(0, len(test_ds), grid_size)
print("image indices:", rnd_inds)
x_grid_test = [test_ds[i][0] for i in rnd_inds]
y_grid_test = [(y_pred[i], y_gt[i]) for i in rnd_inds]
x_grid_test = utils.make_grid(x_grid_test, nrow=4, padding=2)
print(x_grid_test.shape)
plt.rcParams['figure.figsize'] = (10, 5)
imshow(x_grid_test, y_grid_test)
plt.show()
