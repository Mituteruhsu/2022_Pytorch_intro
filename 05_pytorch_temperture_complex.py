import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
# matplotlib inline
features = pd.read_csv('temps.csv')
print(features.head())              # temp_2: 前天的最高氣溫, temp_1: 昨天的最高氣溫

# 顯示 data 維度
print(f'Data 維度: {features.shape}')

# 01-00轉換成時間數據=========================
import datetime
# 01-01分別獲取年月日
years = features['year']
months = features['month']
days = features['day']
# 01-01datatime格式化
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# print(dates[:5])

# 02進行圖形繪製-------------

# 02-1 指定默認格式
plt.style.use('fivethirtyeight')

# 02-2 設置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

# 02-3 標籤值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax2.set_ylabel('Temperature'); ax1.set_title('Max Temp')

# 02-4 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

# 02-5 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

# 02-6 朋友猜測氣溫
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
# plt.show()

# 透過week個別編碼星期幾
features = pd.get_dummies(features)     # 自動生成編碼，因為week中有不同(日字串)，get_dummies會自動每一個日字串個別標籤
features.head(5)
print(features.head(5))

# y (labels['actual'])==============( y = wx + b)
# 製作標籤
labels = np.array(features['actual'])   # 在資料中只有一個實際值(在此假設為Y)

# x (features)==============( y = wx + b)
# 在資料中去掉標籤，就會剩下實際要運行的(在此設為X)
features = features.drop('actual', axis=1)  # 將actual從資料中去除
print(f'Features: \n{features}')
# 名子單獨保存，未來可以使用(所有的表頭名稱)
feature_list = list(features.columns)
print(feature_list)

# 轉換成合適的格式(np的格式，未來可以再轉為tensor)
features = np.array(features)
print(f'Features size: {features.shape}')

# 因為資料在數值上的差異很大，除個位數外尚有十位數或小數點，因此需要標準化
from sklearn import preprocessing   # pip install -U scikit-learn 安裝預先處理模組(標準化)
input_features = preprocessing.StandardScaler().fit_transform(features)     # 將所有features中的數據資料標準化
print(input_features[0])

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 前方預處理完畢 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
'''
建構網路第一步:需要設計網路的結構
y = wx + b
我們的輸入 x (348, 14) 有 14 個特徵, 按照神經網路的做法, 我們需要將輸入的特徵轉換為隱藏層的特徵。
這裡我們先用 128 個特徵來表示隱藏層的特徵 ([348, 14] x [14, 128]) 定義為 w1.

偏置參數該如何定義呢? 對於 w1 來說我們得到了 128 個神經元特徵, 偏置參數的目的是對網路進行微調。 
在這裡大家記住一點就行了, 對於偏置參數來說, 它的 shape 值或者說它的大小永遠是跟你得到的結果是一致的，

我們的結果經過該隱藏層後得到了 128 個特徵, 所以偏置參數也需要有 128 個。
表示我們需要對隱藏層中的 128 個特徵做微調，這裡定義為 b1。

對於回歸任務來說, 我們需要得到的是一個實際的值, 所以我們需要將 128 轉換為 1。
[348, 14] x [14, 128] x [128, 1]
對於 w2 來說, 它的形狀需要設計為 [128, 1] 的矩陣; 同理 b2 應該設計為 1 個特徵.

設計完網路結構後, 下一步我們需要對權重進行初始化操作。
'''

# 建構網路模型
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)
# print(x)
# print(y)


# # 權重參數初始化
weights = torch.randn((14, 128), dtype=float, requires_grad=True)   # ([348, 14] x [14, 128]) 定義為 w1; b1 [128, 1]
biases = torch.randn(128, dtype=float, requires_grad=True)          # b1
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)   # w2 前面連著的是 128 故 w2 [128, 1] b2 [1]
biases2 = torch.randn(1, dtype=float, requires_grad=True)           # b2 最終的結果為 1 個值
# print(weights)
# print(biases)
# print(weights2)
# print(biases2)
learning_rate = 0.001
losses = []
# ************ y = wx + b ************
# 進行迭代
for i in range(1000):
    # 計算隱藏層
    hidden = x.mm(weights) + biases     # w1 + b1
    # 加入激勵函數
    hidden = torch.relu(hidden)         # 除了輸出階層會加上激勵函數外，隱藏層也會加入激勵函數
    # 預測結果
    predictions = hidden.mm(weights2) + biases2 # w2 [128, 1] + b2 
    # 計算損失
    loss = torch.mean((predictions - y) ** 2) # predictions(預測值)- y (actual 真實值) 的平均平方誤差
    losses.append(loss.data.numpy())
    # 列出損失值
    if i % 100 == 0:    # 每100筆列出1次
        print('loss:', loss)
    # 反向傳播計算
    loss.backward()
    # 參數更新
    weights.data.add_(- learning_rate * weights.grad.data)  
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)  
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 每次迭代都需要清空
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
