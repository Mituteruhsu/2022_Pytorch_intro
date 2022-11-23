# 神經網路分類 Mnist 分類
# 網路基本建構與訓練方法，常用函式分析
# torch.nn.funcitonal 模組
# nn.Module 模組

# 讀取 Mnist 數據集 - 會自動下載
# matplotlib inline
# from pathlib import Path
# import requests
# DATA_PATH = Path('data')
# PATH = DATA_PATH / 'mnist'

# PATH.mkdir(parents=True, exist_ok=True)

# URL = 'http://deeplearning.net/data/mnist/'   # 時常down需要找其他的代案
# URL = 'https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/'
# FILENAME = 'mnist.pkl.gz'

# if not (PATH / FILENAME).exists():
#     content = requests.get(URL + FILENAME).content
#     (PATH / FILENAME).open('wb').write(content)

import pickle
import gzip

# 使用 gzip 開啟 gz 的檔案
with gzip.open('mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin1')

from matplotlib import pyplot
import numpy as np
'''
分類時因為最終產生的值是屬於 0 ~ 9 共 10 個概率值。
'''
pyplot.imshow(x_train[0].reshape((28, 28)), cmap = 'gray')
# pyplot.show()         # 顯示圖
# print(x_train.shape)    # (50000, 784) 相當於 50000個樣本數 個別為 w x h x c = 28 x 28 x 1 的圖像，因為是黑白圖通道只有 1，共計784個特徵

# 使用 pytorch 學習後分析手寫的數字
import torch
# 透過torch.tensor轉換成tensor格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train, y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# torch.nn.functional 有很多層和函數
'''
torch.nn.functional 中有很多功能 nn.Module / nn.functional。
一般狀況下, 若模型有可學習的參數則使用 nn.Module(有卷積層、全接層), 其他情況使用 nn.functional(激勵函數、損失函數, 無 w, b)會相對較容易。
'''
import torch.nn.functional as F # functional 通常用 F

'''
分類問題常用的損失函數: 交叉熵(cross-entropy)
在「什麼叫做損失函數跟為什麼是最小化」在分類的狀況下, 通常是希望錯誤率越小越好,所以用錯誤率當損失函數是一個選項,
但實際上我們並不會直接拿分類錯誤率當作損失函數進行最佳化。
'''
loss_func = F.cross_entropy     # 直接調用

def model(xb):
    return xb.mm(weights) + bias

bs = 64
xb = x_train[0:bs]  # m mini-batch from x
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype = torch.float, requires_grad = True)
bs = 64
bias = torch.zeros(10, requires_grad=True)      # 偏置參數

print(loss_func(model(xb), yb))

# 使用 model 來簡化代碼
# 繼承 nn.Module 並且在其構造函數中需要調用 nn.Module 的構造函數
# 不需要再寫反向傳播函數, nn.Module 能利用 autograd 自動實現反向傳播
# Module 中的可學習參數都可以通過 named_parameters() 或者 parameters() 返回迭代器
from torch import nn

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))     # relu 激勵函數
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = Mnist_NN()
print(net)

for name, parameter in net.named_parameters():
    print(name, parameter, parameter.size())

# 建構數據=================================
# 使用 TensorDataset 和 DataLoader 來簡化
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_da = TensorDataset(x_train, y_train)      # 先轉換成 TensorDataset 的格式
train_dl = DataLoader(train_da, batch_size = bs, shuffle=True)  # 用 DataLoader 1 個 batch 下去讀取 64位元

valid_da = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_da, batch_size = bs * 2)            # 128 位元

def get_data(train_da, valid_da, bs):
    return (
        DataLoader(train_da, batch_size=bs, shuffle=True),
        DataLoader(valid_da, batch_size=bs * 2),
    )

# 一般在訓練模型時加上 model.train(), 會正常使用 Batch Normalization 和 Dropout
# 測試的時候一般選擇 model.eval(), 這樣就不會使用 Batch Normalization 和 Dropout

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 訓練函數 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
import numpy as np

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)         # 計算損失

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(steps, model, loss_func, opt, train_dl, valid_dl):      # 訓練方式 steps 一共 delay 多少次, model 定義好的, loss_func, opt 優化器, train_dl, valid_dl
    for step in range(steps):
        model.train()                   # train 與 valid 分開, 訓練每次要 delay 多久
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        
        model.eval()                    # 驗證集上的損失
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('當前 step:' + str(step), '驗證集損失:' + str(val_loss))

from torch import optim

def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 完成定義 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# 取用定義完的函數
train_dl, valid_dl = get_data(train_da, valid_da, bs)   # 透過 DataLoader 取得數據
model, opt = get_model()                                # model = Mnist_NN(), opt= optim.SGD(model.parameters(), lr=0.001)
fit(25, model, loss_func, opt, train_dl, valid_dl)      
