# import torch
# 1.13.0+cpu
# import numpy as np
# numpy-1.23.4

# print(torch.__version__)

# some basic ==============================================================
# x = torch.empty(5, 3) # 生成空的矩陣
# print(x)

# x = torch.rand(5, 4) # 生成隨機矩陣
# print(x)

# x = torch.zeros(5, 4, dtype=torch.long) # 生成矩阵 0 float轉long
# print(x)

# https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/ 資料
# torch 定義了 7種 CPU tensor類型 和 8種 GPU tensor類型

# Data type                     CPU tensor          GPU tensor
# 32-bit floating point         torch.FloatTensor	torch.cuda.FloatTensor
# 64-bit floating point         torch.DoubleTensor	torch.cuda.DoubleTensor
# 16-bit floating point         N/A                 torch.cuda.HalfTensor
# 8-bit integer (unsigned)      torch.ByteTensor	torch.cuda.ByteTensor
# 8-bit integer (signed)        torch.CharTensor	torch.cuda.CharTensor
# 16-bit integer (signed)       torch.ShortTensor	torch.cuda.ShortTensor
# 32-bit integer (signed)	    torch.IntTensor     torch.cuda.IntTensor
# 64-bit integer (signed)	    torch.LongTensor	torch.cuda.LongTensor

# x = torch.tensor([5, 3]) # 列表[5, 3]轉換成tensor的矩陣
# print(x)
# x = x.new_ones([5, 3], dtype=torch.double) # 構造全為1的數組
# print(x)
# x = torch.rand_like(x, dtype=torch.float) # 生成相同維度的隨機矩陣
# print(f'隨機生成與new_ones同維度矩陣 x:\n{x}')
# y = torch.rand(5, 3) # 生成隨機的矩陣
# print(f'隨機生成維度矩陣 y:\n {y}')
# print(f'x + y:\n {x + y}')
# print(torch.add(x, y)) # 進行矩陣的相加操作類似 x + y

# # 索引======================================================
# print(x[:, 0])
# print(x[:, 1])
# print(x[:, 2])

# # view 可以用來改變矩陣維度==================================
# x = torch.randn(4, 4)
# print(x)
# y = x.view(16)
# print(y)
# z = x.view(-1, 8)
# print(z)
# # tensor 與 numpy 間轉換====================================
# a = torch.ones(5)  # tensor 創造全部1的數組 5個
# print(a)
# b = a.numpy()  # tensor 的數組轉換成 numpy 格式
# print(b)

# import numpy as np
# numpy-1.23.4
# a = np.ones([3, 2])
# print(a)
# b = torch.from_numpy(a) # 將 numpy 格式轉換tensor格式
# print(b)

'''
張量（英語：Tensor）是一個可用來表示在一些向量、純量和其他張量之間的線性關係的多線性函數，這些線性關係的基本例子有內積、外積、線性映射以及笛卡兒積。其坐標在 {\displaystyle n}n  維空間內，有  {\displaystyle n^{r}}n^r個分量的一種量，其中每個分量都是坐標的函數，而在坐標變換時，這些分量也依照某些規則作線性變換。{\displaystyle r}r稱為該張量的秩或階（與矩陣的秩和階均無關係）。

在同構的意義下，第零階張量（{\displaystyle r=0}r=0）為純量，第一階張量（{\displaystyle r=1}r=1）為向量， 第二階張量（{\displaystyle r=2}r=2）則成為矩陣。例如，對於3維空間，{\displaystyle r=1}r=1時的張量為此向量：{\displaystyle \left(x,y,z\right)^{\mathrm {T} }}\left( x,y,z \right)^\mathrm{T}。由於變換方式的不同，張量分成「協變張量」（指標在下者）、「逆變張量」（指標在上者）、「混合張量」（指標在上和指標在下兩者都有）三類。

在數學裡，張量是一種幾何實體，或者說廣義上的「數量」。張量概念包括純量、向量和線性算子。張量可以用坐標系統來表達，記作純量的數組，但它是定義為「不依賴於參照系的選擇的」。張量在物理和工程學中很重要。例如在擴散張量成像中，表達器官對於水的在各個方向的微分透性的張量可以用來產生大腦的掃描圖。工程上最重要的例子可能就是應力張量和應變張量了，它們都是二階張量，對於一般線性材料他們之間的關係由一個四階彈性張量來決定。

雖然張量可以用分量的多維數組來表示，張量理論存在的意義在於進一步說明把一個數量稱為張量的涵義，而不僅僅是說它需要一定數量的有指標索引的分量。特別是，在坐標轉換時，張量的分量值遵守一定的變換法則。張量的抽象理論是線性代數分支，現在叫做多重線性代數。

張量在物理學中提供了一個簡明的數學框架用來描述和解決力學（應力、彈性、流體力學、慣性矩等）、電動力學（電磁張量、馬克士威張量、介電常數、磁化率等）、廣義相對論（應力-能量張量、曲率張量等）物理問題。在應用中，數學家通常會研究在物體的不同點之間的張量變化; 例如，一個物體內的應力可能因位置不同而改變。這就引出了張量場的概念。在某些領域，張量場十分普遍以至於它們通常被簡稱為「張量」。
'''
# # 00 Basic use ===================================
# # 00-1 創建空的矩陣(tensor指的是任何維度的矩陣)------
# x = torch.empty(5, 3)
# print(x)

# # 00-2 torch.rand(生成隨機矩陣)---------------------
# x = torch.rand(5,3)
# print(x)

# # 00-3 torch.zeros(初始化一個全為零的矩陣)-----------
# # dtype=torch.long 或使用 .long() 是將float 類型的tensor 轉換為long
# x = torch.zeros(5,3, dtype=torch.long)
# print(x)
# x = torch.zeros(5, 3)
# x = x.long()
# print(x)

# # 00-4 torch.tensor(傳入數據)-----------------------
# import torch
# x = torch.tensor([5.5, 3])
# # print(x)
# x = x.new_ones(5, 3, dtype=torch.double)
# # print(x)
# x = torch.randn_like(x, dtype=torch.float)
# # print(x)
# tensorsize = x.size() # 可顯示矩陣大小
# # print(tensorsize)

# # 01 Pytorch tensor(基本計算方法)===================
# # 01-1 torch.add(矩陣相加)--------------------------
# import torch
# x = torch.tensor([5.5, 3])
# x = x.new_ones(5, 3, dtype=torch.double)
# x = torch.randn_like(x, dtype=torch.float)
# y = torch.rand(5, 3)
# print(f'矩陣 x: \n{x}')
# print(f'矩陣 y: \n{y}')
# print(f'矩陣 x + y: \n{x + y}')
# print(f'使用 add(x, y) \n {torch.add(x, y)}')

# # 01-2 索引-----------------------------------------
# import torch
# x = torch.tensor([5.5, 3])
# x = x.new_ones(5, 3, dtype=torch.double)
# x = torch.randn_like(x, dtype=torch.float)
# print(x)
# print(x[:, 0])
# print(x[:, 1])
# print(x[:, 2])

# # 01-3 view 可以用來改變矩陣維度----------------------
# import torch
# x = torch.randn(4, 4)
# print(x)
# y = x.view(16)
# print(y)
# z = x.view(-1, 8) # -1 代表自動計算 4*4=16, 16/8= 2 (故 -1 會顯示 2)
# print(z)
# print(x.size(), y.size(), z.size())

# # 01-4 tensor 與 numpy 間轉換-------------------------
# import torch
# a = torch.ones(5)  # tensor 創造全部1的數組 5個
# print(a)
# b = a.numpy()  # tensor 的數組轉換成 numpy 格式
# print(f'tensor to numpy: \n{b}')

# import numpy as np
# # numpy-1.23.4
# a = np.ones([3, 2])
# print(f'numpy: \n{a}')
# b = torch.from_numpy(a) # 將 numpy 格式轉換tensor格式
# print(f'numpy to => \n{b}')

# # 02 Pytorch (反向傳播)===============================
# # 02-1 requires_grad ---------------------------------
# import torch
# # # *******method 1 (建議使用，一行比較好讀)*******
# x = torch.randn(3, 4, requires_grad=True)    # True 為 1, 加入requires_grad 指定需要的確認值x
# print(f'x = \n{x}')
# # # *******method 2*******
# # x = torch.randn(3, 4)
# # x.requires_grad=True
# # print(x)
# # ************************

# b = torch.randn(3, 4, requires_grad=True)    # True 為 1, 加入requires_grad 指定需要的確認值b
# t = x + b                                    # 雖然未加入requires_grad 但是pytorch會自動求導 x, b, t 的關聯
# print(f't = x + b \n{t}')
# y = t.sum()
# print(f't sum \n{y}')
# y.backward()                                 # x + b -> t, t sum -> y 從 y backward
# print(b.grad)
# print(x.requires_grad, b.requires_grad, t.requires_grad)

# # 02-2 requires_grad 計算流程--------------------------
# import torch
# x = torch.rand(1)
# b = torch.rand(1, requires_grad=True)
# w = torch.rand(1, requires_grad=True)
# y = w * x
# z = y + b
# print(x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad, z.requires_grad)
# print(x.is_leaf, w.is_leaf, b.is_leaf, y.is_leaf, z.is_leaf)

# # *****反向傳播計算*****
# z.backward(retain_graph=True)   # 梯度如果不清空會累加
# wg = w.grad
# bg = b.grad
# print(wg)
# print(bg)

'''
什麼是葉子點張量?葉子點張量通常需要滿足兩個條件。
1. 葉子節點張量是由用戶直接創建的張量,非由某個Function通過計算得到的張量。
2. 葉子節點張量的 requires_grad 的屬性必須為True。
'''
# 判斷一個張量是不是葉子節點,可以通過它的屬性is_leaf來檢視.
# 按照慣例, 所有屬性 requires_grad=False 的張量是葉子節點(即:葉子張量、葉子節點張量).
# 對於屬性requires_grad=True的張量可能是葉子節點張量也可能不是葉子節點張量而是中間節點(中間節點張量).
# 如果該張量的屬性requires_grad=True, 而且是用於直接建立的, 它的屬性grad_fn=None,那麼它就是葉子節點.
# 如果該張量的屬性requires_grad=True, 但是它不是使用者直接建立的, 而是由其他張量經過某些運算操作產生的, 那麼它就不是葉子張量,
# 而是中間節點張量, 並且它的屬性 grad_fn 不是 None,
# 比如:grad_fn=<MeanBackward0>,
# 這表示該張量是通過 torch.mean() 運算操作產生的,是中間結果,所以是中間節點張量,不是葉子節點張量.

# 一個張量的屬性 requires_grad 用來指示在反向傳播時, 是否需要為這個張量計算梯度.
# 如果這個張量的屬性 requires_grad=False, 那麼就不需要為這個張量計算梯度, 也就不需要為這個張量進行優化學習.
# 在PyTorch的運算操作中, 如果參加這個運算操作的所有輸入張量的屬性 requires_grad 都是False的話,
# 那麼這個運算操作產生的結果, 即輸出張量的屬性 requires_grad也是False, 否則是True.
# 輸入的張量只要有一個需要求梯度 (屬性 requires_grad=True),
# 那麼得到的結果張量也是需要求梯度的 (屬性requires_grad=True).
# 只有當所有的輸入張量都不需要求梯度時, 得到的結果張量才會不需要求梯度.

# 對於屬性 requires_grad=True 的張量, 在反向傳播時, 會為該張量計算梯度.
# 但是 pytorch 的自動梯度機制不會為中間結果儲存梯度, 只會為葉子節點計算的梯度儲存起來, 
# 儲存到該葉子節點張量的屬性 grad 中, 不會在中間節點張量的屬性 grad 中儲存這個張量的梯度,
# 這是出於對效率的考慮, 中間節點張量的屬性 grad 是None.
# 如果使用者需要為中間節點儲存梯度的話, 可以讓這個中間節點呼叫方法 retain_grad(), 這樣梯度就會儲存在這個中間節點的grad屬性中.

# # 03 Pytorch (線性回歸圖形)===============================
# # y = 2x + 1 <= 做為本次的設定
# # 分兩組數組 x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y= [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
import numpy as np
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
print(x_values)
# print(x_train)
print(x_train.shape)
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_values)
# print(y_train)
print(y_train.shape)

# # # 03-1 pytorch (線性回歸模型)--------------------------
# # nn.Linear()：用於設定網路中的全連線層，需要注意的是全連線層的輸入與輸出都是二維張量
import torch
import torch.nn as nn

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
# print(model)
# # 指定 (參數) 與 (損失函數)
# epochs = 1000           # 學習次數
# learning_rate = 0.01    # 學習率
# # 優化器 SGD 除SGD外還有其他的 SGD 需要優化什麼? model.parameters(), 並且置入學習率 lr = learning_rate
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()    # 損失函數
# # # 訓練模型-------------------------------------------------
# for epoch in range(epochs):
#     epoch += 1
#     # 轉型成 tensor 格式
#     inputs = torch.from_numpy(x_train)
#     labels = torch.from_numpy(y_train)

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
    
# 測試模型預測結果
# 使用.data.numpy() 轉換成numpy格式
# predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
# print(predicted)

# 模型的保存與讀取
# torch.save(model.state_dict(), 'model.pkl')
model.load_state_dict(torch.load('model.pkl'))

# 測試模型預測結果
# 使用.data.numpy() 轉換成numpy格式
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)
