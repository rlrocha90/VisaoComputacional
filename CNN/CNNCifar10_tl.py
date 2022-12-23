import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms

batch_size = 50

# Dados de Treinamento
train_data_transform = transforms.Compose([
    transforms.Resize(224), # ImageNet considera entrada 224x224 - Upsampling
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=train_data_transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

val_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_set = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=val_data_transform)
val_order = torch.utils.data.DataLoader(val_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, loss_function, optimizer, data_loader):
    # Define modelo em modo treinamento
    model.train()
    current_loss = 0.0
    current_acc = 0
    # iterar em todo o conjunto de treinamento
    for i, (inputs, labels) in enumerate(data_loader):
        # Envia o dado para GPU, se disponível
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zera os parâmetros do gradiente
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # etapa Forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
            # etapa backward
            loss.backward()
            optimizer.step()
        # estatísticas
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
        total_loss = current_loss / len(data_loader.dataset)
        total_acc = current_acc.double() / len(data_loader.dataset)
        print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, loss_function, data_loader):
    # define o modelo em modo de teste
    model.eval()
    current_loss = 0.0
    current_acc = 0
    # itera sobre o conjunto de validação
    for i, (inputs, labels) in enumerate(data_loader):
        # envia dados para GPU, se disponível
        inputs = inputs.to(device)
        labels = labels.to(device)
        # etapa forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
        # etapa statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


# Primeiro cenário: Transferência de Aprendizado usando a rede prétreinada como extrator de características
def tl_feature_extractor(epochs=3):
    # carrega o modelo prétreinado
    model = torchvision.models.resnet18(pretrained=True)
    # congelar o modelo
    for param in model.parameters():
        param.requires_grad = False
    # novas camadas construídas precisam de requires_grad=True por padrão
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    # transfere para GPU (if available)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    # apenas os parâmetros da camada final são otimizados
    optimizer = optim.Adam(model.fc.parameters())
    # treinamento
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, val_order)


# Abordagem juste fino (Fine-tunning)
def tl_fine_tuning(epochs=3):
    # carrega o modelo prétreinado
    model = models.resnet18(pretrained=True)
    # troca a última camada
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    # envia modelo para GPU (se disponível)
    model = model.to(device)
    # função custo
    loss_function = nn.CrossEntropyLoss()
    # otimiza todos os parâmetros...
    optimizer = optim.Adam(model.parameters())
    # treinamento
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, val_order)


tl_fine_tuning(epochs=5)
tl_feature_extractor(epochs=5)

