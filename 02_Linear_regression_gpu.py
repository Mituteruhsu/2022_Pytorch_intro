# # Pytorch GPU(線性回歸圖形 Linear regression)===============================
# # y = 2x + 1 <= 做為本次的設定
# # 分兩組數組 x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y= [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
import torch
import torch.nn as nn
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
print(f'x values = {x_values}')
# print(x_train)
print(f'x value shape = {x_train.shape}')
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(f'y values = {y_values}')
# print(y_train)
print(f'y value shape = {y_train.shape}')

# # 架設模組=====================================================================
# # nn.Linear()：用於設定網路中的全連線層，需要注意的是全連線層的輸入與輸出都是二維張量
class LinerRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinerRegressionModel, self).__init__()
        self.linear= nn.Linear(input_dim, output_dim)   # 全連接層

    def forward(self, x):                               # 用到的層
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinerRegressionModel(input_dim, output_dim)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# criterion = nn.MSELoss()    # 損失函數
# # print(model)
# # # 指定 (參數) 與 (損失函數)
# epochs = 1000           # 學習次數
# learning_rate = 0.01    # 學習率
# # 優化器 SGD 除SGD外還有其他的 SGD 需要優化什麼? model.parameters(), 並且置入學習率 lr = learning_rate
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()    # 損失函數
# # # 訓練模型-------------------------------------------------
# for epoch in range(epochs):
#     epoch += 1
#     # 轉型成 tensor 格式
#     inputs = torch.from_numpy(x_train).to(device)
#     labels = torch.from_numpy(y_train).to(device)

#     # 梯度每疊代一次就要清零
#     optimizer.zero_grad()
#     # 正向傳播
#     outputs = model(inputs)
#     # 計算損失
#     loss = criterion(outputs, labels)
#     # 反向傳播
#     loss.backward()
#     # 更新權重參數
#     optimizer.step()
#     if epoch % 50 == 0:         # 每隔 50 次print一次
#         print('epoch {}, loss {}'.format(epoch, loss.item()))
    
# # 模型的保存與讀取
# # *****保存*****
# torch.save(model.state_dict(), 'model_gpu.pkl')
# # *****讀取*****
model.load_state_dict(torch.load('model_gpu.pkl'))

# 測試模型預測結果
# 使用.data.numpy() 轉換成numpy格式
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)

