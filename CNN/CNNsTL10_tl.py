from torchvision import datasets
import torchvision.transforms as transforms
import os

path2data="./data"

if not os.path.exists(path2data):
    os.mkdir(path2data)
data_transformer = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.STL10(path2data, split='train', download=True, transform=data_transformer) #5,000 imagens 3*96*96
print(train_ds.data.shape)

import collections

# Contagem de número de imagens em cada categoria no train_ds
y_train = [y for _, y in train_ds]
counter_train = collections.Counter(y_train)
print(counter_train)

# Carregar os dados de teste em test0_ds
test0_ds = datasets.STL10(path2data, split='test', download=True, transform=data_transformer)
print(test0_ds.data.shape)

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

# Contar o número de imagens por classe em val_ds e test_ds
import collections
import numpy as np

y_test = [y for _, y in test_ds]
y_val = [y for _, y in val_ds]

counter_test = collections.Counter(y_test)
counter_val = collections.Counter(y_val)
print(counter_test)
print(counter_val)

# Mostrar alguns exemplos de imagens, em train_ds
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def show(img, y=None, color=True):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr)
    if y is not None:
        plt.title("label: " + str(y))


grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print("image indices:", rnd_inds)

x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]
x_grid = utils.make_grid(x_grid, nrow=4, padding=1)
print(x_grid.shape)

plt.figure(figsize=(10, 10))
#show(x_grid, y_grid)

# Mostrar algumas imagens do conjunto val_ds
np.random.seed(0)

grid_size = 4
rnd_inds = np.random.randint(0, len(val_ds), grid_size)
print("image indices:", rnd_inds)
x_grid = [val_ds[i][0] for i in rnd_inds]
y_grid = [val_ds[i][1] for i in rnd_inds]
x_grid = utils.make_grid(x_grid, nrow=4, padding=2)
print(x_grid.shape)
plt.figure(figsize=(10, 10))
show(x_grid, y_grid)

# Calcular média e desvio padrão do conjunto train_ds
import numpy as np

meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_ds]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])
print(meanR, meanG, meanB)
print(stdR, stdG, stdB)

# Definir algumas transformações, associadas ao calculado anteriormente
train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # Aumento de dados
    transforms.RandomVerticalFlip(p=0.5), # Aumento de dados
    transforms.ToTensor(),
    transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])]) # transformar a imagem para média 0 e variância 1
test0_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
])

# Transformar os dados
train_ds.transform = train_transformer
test0_ds.transform = test0_transformer

# Mostrar imagens transformadas
import torch
np.random.seed(0)
torch.manual_seed(0)

grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print("image indices:", rnd_inds)
x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]
x_grid = utils.make_grid(x_grid, nrow=4, padding=2)
print(x_grid.shape)

plt.figure(figsize=(10, 10))
show(x_grid, y_grid)

# Criar Dataloaders a partiir dos dados train_ds e val_ds
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

# Obter um "batch" de dados a partir de train_dl
for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break

# Obter um "batch" de dados a partir de val_dl
for x, y in val_dl:
    print(x.shape)
    print(y.shape)
    break

# CONSTRUÇÃO DO MODELO
# Construir o modelo, sem a definição de um modelo customizado.

# IMPORTANTE: A biblioteca torchvision tem múltiplas implementações "estado da arte" para modelos deep learning para
# classificação de imagem. Exemplos: AlexNet, VGG, ResNet, SqueezeNet, DenseNet, Inception, GoogleNet e ShuffleNet.
# São modelos treinamento no dataset ImageNet que conta com mais de 14 milhões de imagens em 1000 classes.

# Para Transfer Learning, pode-se usar o modelo com pesos iniciados aleatórios ou os pesos pré treinados.

# CARREGAR UM MODELO PRÉ TREINADO
# Importar Resnet-18 com pesos aleatórios

from torchvision import models
import torch

model_resnet18 = models.resnet18(weights=None)
print(model_resnet18)

# Trocar a camada de saída para 10 classes
from torch import nn

num_classes = 10
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_resnet18.to(device)

from torchsummary import summary

summary(model_resnet18, input_size=(3, 224, 224), device=device.type)

# Visualização dos Filtros da primeira camada
for w in model_resnet18.parameters():
    w = w.data.cpu()
    print(w.shape)
    break

# Normalizar os pesos
min_w = torch.min(w)
w1 = (-1/(2*min_w))*w + 0.5
print(torch.min(w1).item(), torch.max(w1).item())

grid_size = len(w1)
x_grid = [w1[i] for i in range(grid_size)]
x_grid = utils.make_grid(x_grid, nrow=8, padding=1)
print(x_grid.shape)
plt.figure(figsize=(10, 10))
show(x_grid)

# CARREGAR UM MODELO PRÉ TREINADO
# Importar Resnet-18 com pesos pré-treinados

from torchvision import models
import torch

resnet18_pretrained = models.resnet18(pretrained=True)
num_classes = 10
num_ftrs = resnet18_pretrained.fc.in_features
resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18_pretrained.to(device)

# Visualização dos Filtros da primeira camada
for w in resnet18_pretrained.parameters():
    w = w.data.cpu()
    print(w.shape)
    break

# Normalizar os pesos
min_w = torch.min(w)
w1 = (-1/(2*min_w))*w + 0.5
print(torch.min(w1).item(), torch.max(w1).item())

grid_size = len(w1)
x_grid = [w1[i] for i in range(grid_size)]
x_grid = utils.make_grid(x_grid, nrow=8, padding=1)
print(x_grid.shape)
plt.figure(figsize=(10, 10))
show(x_grid)

# Definir a função "loss" - Como otimizar
loss_func = nn.CrossEntropyLoss(reduction="sum")

torch.manual_seed(0)

n, c=4, 5
y = torch.randn(n, c, requires_grad=True)
print(y.shape)

loss_func = nn.CrossEntropyLoss(reduction="sum") # Soma dos valores de loss por "batch of data"
target = torch.randint(c, size=(n,))
print(target.shape)

loss = loss_func(y, target)
print(loss.item())

# Gradiente da função loss com respeito a y(saída)
loss.backward()
print(y.data)

# Definir o otimizador e a taxa de aprendizado
from torch import optim

opt = optim.Adam(model_resnet18.parameters(), lr=1e-4)

# Mostra o valor do Learning Rate no momento atual
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


current_lr = get_lr(opt)
print('current lr={}'.format(current_lr))

from torch.optim.lr_scheduler import CosineAnnealingLR

lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-5)
for i in range(10):
    lr_scheduler.step()
    print("epoch %s, lr: %.1e" %(i, get_lr(opt)))

# Treinar o Modelo

# Definição de contagem de predições corretas
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# Definição de função para computar valores "loss" e métricas de desempenho
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        if sanity_check is True:
            break

    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

# Extração dos parâmetros
def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history = {
        "train": [],
        "val": [],
    }
    metric_history = {
        "train": [],
        "val": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

        lr_scheduler.step()
        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" % (train_loss, val_loss, 100 * val_metric))
        print("-" * 10)

    # Retorna o melhor modelo
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

# Treinar o modelo chamando a função - Pesos aleatórios

import copy
loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model_resnet18.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(opt, T_max=5, eta_min=1e-6)

os.makedirs("./models", exist_ok=True)
params_train = {
    "num_epochs": 3,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": "./models/resnet18.pt",
}
model_resnet18, loss_hist, metric_hist = train_val(model_resnet18, params_train)

num_epochs = params_train["num_epochs"]
plt.figure()
plt.title("Train-Val Loss")
plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs+1), loss_hist["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

plt.figure()
plt.title("Train-Val Accuracy")
plt.plot(range(1, num_epochs+1), metric_hist["train"], label="train")
plt.plot(range(1, num_epochs+1), metric_hist["val"], label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# Treinamento com pesos pré treinado
import copy
loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(resnet18_pretrained.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(opt, T_max=5, eta_min=1e-6)
params_train={
    "num_epochs": 3,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": "./models/resnet18_pretrained.pt",
}
resnet18_pretrained, loss_hist, metric_hist = train_val(resnet18_pretrained, params_train)

plt.figure()
num_epochs = params_train["num_epochs"]
plt.title("Train-Val Loss")
plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs+1), loss_hist["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

plt.figure()
plt.title("Train-Val Accuracy")
plt.plot(range(1, num_epochs+1), metric_hist["train"], label="train")
plt.plot(range(1, num_epochs+1), metric_hist["val"], label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

