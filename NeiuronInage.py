import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

#Проверка папки и ее содержимое
# mypath = "EsperssoMachine"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(onlyfiles)

# plt.rcParams['figure.figsize'] = 14, 6

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(16 * 16 * 16, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x
# Путь к папке с фотографиями
# data_dir = "C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/EsperssoMachine"

# normalize_transform = transforms.Compose([
#     transforms.Resize((960, 1280)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# train_dt = torchvision.datasets.ImageFolder(
#     root=data_dir,
#     transform=normalize_transform)
# test_dt = torchvision.datasets.ImageFolder(
#     root=data_dir,
#     transform=normalize_transform)

# batch_size = 128
# train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size)

# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# plt.imshow(np.transpose(torchvision.utils.make_grid(
#   images[:22], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
# plt.axis('off')
# plt.show()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# # Определение модели
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(16 * 16 * 16, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x
# # Путь к папке с фотографиями
# data_dir = "C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/EsperssoMachine"
# # Преобразование данных
# data_transforms = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# # Загрузка данных
# train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
# # Создание загрузчика данных
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# # Инициализация модели
# model = MyModel()
# # Определение функции потерь и оптимизатора
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # Обучение модели
# for epoch in range(10):
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 100 == 99:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0
#         else:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0

# print("Training finished!")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# # Определение модели
# class MyModel(nn.Module):
#     def __init__(self, num_classes):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.LeakyReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(16 * 16 * 4, num_classes)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x
# # Путь к папке с фотографиями
# data_dir = "C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/EsperssoMachine"
# # Преобразование данных
# normalize_transform = transforms.Compose([
#     transforms.Resize((16, 16)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# # Загрузка данных
# train_dataset = datasets.ImageFolder(data_dir, transform=normalize_transform)
# print(train_dataset)
# # Создание загрузчика данных
# batch_size = 128
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # Инициализация модели
# num_classes = len(train_dataset.classes)
# print("numnum", num_classes)
# model = MyModel(num_classes)
# # Определение функции потерь и оптимизатора
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # Обучение модели
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         print(loss)
# # Классификация введенной фотографии
# input_image = Image.open("C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/photo_2023-06-25_16-32-28.jpg")
# input_image = normalize_transform(input_image)
# input_image = input_image.unsqueeze(0)  # добавление размерности пакета
# model.eval()
# with torch.no_grad():
#     output = model(input_image)
# predicted_class = torch.argmax(output).item()
# print(predicted_class)
# class_labels = train_dataset.classes
# if predicted_class == 0:
#     print("Это то фото")
# else:
#     print("Это не то фото")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
# Путь к папке с фотографиями
data_dir = "C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/EsperssoMachine"
# Преобразование данных
normalize_transform = transforms.Compose([
    transforms.Resize((860, 680)),  # Изменяем размер изображений до 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Загрузка данных
train_dataset = datasets.ImageFolder(data_dir, transform=normalize_transform)
print(train_dataset)
# Создание загрузчика данных
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Загрузка предобученной модели ResNet-18
model = models.resnet152(pretrained=True)
# Заморозка параметров модели
for param in model.parameters():
    param.requires_grad = False
# Замена последнего полносвязного слоя модели
num_features = model.fc.in_features
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(num_features, num_classes)
# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
# Обучение модели
num_epochs = 7
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss)
# Классификация введенной фотографии
input_image = Image.open("C:/Users/shayq/OneDrive/Рабочий стол/Питон/BotCleanCheck/photo_2023-06-25_14-10-22.jpg")
input_image = normalize_transform(input_image)
input_image = input_image.unsqueeze(0)  # добавление размерности пакета
model.eval()
with torch.no_grad():
    output = model(input_image)
predicted_class = torch.argmax(output).item()
print(predicted_class)
class_labels = train_dataset.classes
if predicted_class == 0:
    print("Это то фото")
else:
    print("Это не то фото")