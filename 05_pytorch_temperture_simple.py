# P.30
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
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),   # 第一層:輸入至隱藏 層
    torch.nn.Sigmoid(),                         # 激勵函數
    torch.nn.Linear(hidden_size, output_size)   # 全接層:隱藏至輸出 層
)
cost = torch.nn.MSELoss(reduction='mean')       # 'mean' 名均 相當於計算損失predictions(預測值)- y (actual 真實值) 的平均平方誤差
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)    # 取代優化器參數更新與迭代歸零 Adam、SGD等方式，但是Adam 能夠動態的去調整

# 訓練網路
losses = []
for i in range(1000):
    batch_loss = []
    # M1N1-Batch方法進行訓練
    for start in range(0, len(input_features), batch_size):     # 非取全部數據
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype = torch.float, requires_grad = True)         # x tensor 中有取 start / end 因為batch_size = 16 因此每一次都會取16個數據
        yy = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)                 # y tensor 中同上取16個數據
        prediction = my_nn(xx)      # 預測則是前向傳播，在此 my_nn(xx) 已經做了每一層的定義
        loss = cost(prediction, yy) # 放入預測值與真實值
        optimizer.zero_grad()       # 每次梯度會歸零
        loss.backward(retain_graph=True)    # 反向傳播
        optimizer.step()                    # 反向傳播同樣更新
        batch_loss.append(loss.data.numpy())    # 重複執行
    # 每100次印出損失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 預測訓練結果
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()
# print(predict)

# 日期格式轉換

# 01-01datatime格式化
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 創建一個表格來儲存日期和其對應的標籤數值
true_data = pd.DataFrame(data = {'date':dates, 'actual':labels})

# 創建一個用來放日期和對應模型的預測值
years = features[:, feature_list.index('year')]
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predict.reshape(-1)})

# 真實值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')    # 'b-' 指的是藍色線

# 預測值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')  # 'ro' 紅色點
plt.xticks(rotation = '60')
plt.legend()

# 圖名
plt.xlabel('Date');plt.ylabel('Maximum Temperture (F)');plt.title('Actual and Predicted Values')
plt.show()