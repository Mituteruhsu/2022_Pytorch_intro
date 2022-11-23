import os
import time
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models, datasets
import imageio
import warnings
import random
import sys
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.025),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    
    ]),
    'valid': transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
# print(image_datasets)

# print(dataloaders)

# print(dataset_sizes)

# 讀取標籤對應的實際名子
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# print(cat_to_name)

def im_convert(tensor):
    '''展示數據'''
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(dataloaders['valid'])
inputs, classes = dataiter.__next__()

for idx in range(columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    plt.imshow(im_convert(inputs[idx]))
# plt.show()


model_name = 'resnet'       # 有多項 model 可選擇 ['resnet', 'alexne', 'vgg', 'squeezenet', 'densenet', 'inception']
feature_extract = True      # 是否使用訓練好的特徵

# 是否使用 GPU 訓練
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available! Training on GPU...')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_parameter_requires_grad(model, featrue_extracting):
    if featrue_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft = models.resnet152()       # 加入模型 resnet152
# print(model_ft)                     # 印出獲得這些數據的流程

#參考 Pytorch 官網的例子使用 Model=====================================================
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Model set ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet':
        '''Resnet152
        '''
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102), 
                                    nn.LogSoftmax(dim=1))
        input_size = 224
    
    elif model_name == 'alexnet':
        '''Alexnet
        '''
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'vgg':
        '''VGG11_bn
        '''
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == 'squeezenet':
        '''Squeezenet
        '''
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    
    elif model_name == 'densenet':
        '''Densenet
        '''
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == 'inception':
        '''Inception v3
        Be careful, expects (299, 299) sized images and has auxiliary output
        '''
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print('Invalid model name, exiting...')
        exit()
    
    return model_ft, input_size
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Model set ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# 訓練所需的層=================================================================================

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
# 使用GPU計算
model_ft = model_ft.to(device)
# 模型保存
filename = 'checkpoint.pth'
# 是否訓練所有層
params_to_update = model_ft.parameters()
print('Params to learn:')
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print('\t', name)

# print(model_ft)       # 神經網路的架構

# 優化器的設置=============================================================================================
optimizer_ft = optim.Adam(params_to_update, lr = 1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma = 0.1) # 學習率每7個epoch就衰減成原來的 1/10
# 最後一層已經 LogSoftmax(), 所以不能使用 nn.CrossEntropyLoss()來計算, nn.CrossEntropyLoss()相當於 logSoftmax() 和 nn.NLLLoss() 的整合
criterion = nn.NLLLoss()

# 訓練模組
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()
    best_acc = 0            # 驗證保存最好的
    '''
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']    
    '''
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())  # 學習時拿取最好的 model

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 訓練和驗證
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()   # 訓練
            else:
                model.eval()    # 驗證
            
            running_loss = 0.0
            running_corrects = 0

            # 運用遍歷將數據都取一遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有訓練的時候計算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:       # resnet 執行點在這裡
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                    # 訓練階段與更新權重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 計算損失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                   'state_dict': model.state_dict(),
                   'best_acc': best_acc,
                   'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

# 訓練完後用最好的一次當作模型的最終結果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

# 執行-開始進行訓練
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=5, is_inception=(model_name=='inception'))

# 在訓練所有層
for param in model_ft.parameters():
    param.requires_grad = True
    
# 再訓練所有的參數, 學習率調小一點
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 損失函數
criterion = nn.NLLLoss()

# Load the checkpoint
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
# model_ft.class_to_idx = checkpoint['mapping']

# 執行
# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=5, is_inception=(model_name=='inception'))

# 測試網路效果
# 輸入圖像進行測試, 看回傳的結果
# probs, classes = predict(impage_path, model)
# print(probs)
# print(classes)
# 實際做法

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU 模式
model_ft = model_ft.to(device)

# 保存文件的名子
filename = 'seriouscheckpoint.pth'

# 載入模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

# 測試數據預處理
def process_image(image_path):
    img = Image.open(image_path)
    # Resize, thumbnail方法只能進行縮小, 所以先判斷
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 操作
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # 相同的預處理方法
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])  # provide mean
    std = np.array([0.229, 0.224, 0.225])   # provide std
    img = (img - mean)/std

    # 注意顏色通道需放在第一個位置
    img = img.transpose((2, 0, 1))

    return img

def imshow(image, ax=None, title=None):
    '''展示數據'''
    if ax is None:
        fig, ax = plt.subplots()
    
    # 顏色還原
    image = np.array(image).transpose(1, 2, 0)

    # 預處理還原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])   # provide std
    image = std * image + mean    
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax

image_path = 'image_06621.jpg'
img = process_image(image_path)
imshow(img)
# plt.show()    # 這裡會將前面設定好的所有圖片也顯示出來
print(img.shape)

# 得到一個 batch 的測試數據
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.__next__()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

# 獲取 output 的數據值, 對每一個 batch 中每一個數據得到其屬於各種類別的可能性
print(output.shape)

# 獲取概率最大的那一個
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
print(preds)

# 展示預測結果==========================================
fig = plt.figure(figsize=(20, 20))
columns=4
rows=2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title('{} ({})'.format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                color=('green' if cat_to_name[str(preds[idx])]==cat_to_name[str(labels[idx].item())] else 'red'))

# plt.show() # 顯示到此為止的所有的圖片