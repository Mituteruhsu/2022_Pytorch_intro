import time
import copy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# 先定義參數
input_size = 28         # 圖像的總尺寸 28 x 28
num_classes = 10        # 標籤的種類數
num_epochs = 3          # 訓練的總循環週期
batch_size = 64         # 一個批次的大小, 共64張圖片

# 訓練集
train_dataset = datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 測試集
test_dataset = datasets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

# 建構 batch 數據
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

# 卷積神經網路構建
# 一般卷積層, relu層, 池化層可寫成一整套
# 卷積最後還是一個特徵圖, 需要把圖轉換成向量才能做分類或者回歸
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

# 準確率做為評估標準
def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

# 訓練網路模型======================================================
# 建立 CNN net
net = CNN()
# 損失函數
criterion = nn.CrossEntropyLoss()
# 優化器
optimizer = optim.Adam(net.parameters(), lr = 0.001)
# 訓練循環
for epoch in range(num_epochs):
    train_rights = []           # 將當前的 epoch 結果保留

    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(output, target)

        if batch_idx % 100 == 0:

            net.eval()
            val_rights = []

            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)
            
            # 準確率計算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('當前 epoch: {} [{}/{} ({:.0f}%)]\t損失: {:.6f}\t訓練集準確率: {:.2f}%\t測試集準確率: {:.2f}%'.format(
                epoch, batch_idx*batch_size, len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                loss.data,
                100.*train_r[0].numpy() / train_r[1],
                100.*val_r[0].numpy() / val_r[1]))