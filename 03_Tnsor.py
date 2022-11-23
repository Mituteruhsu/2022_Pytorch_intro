# 0. scalar 純量(沒有方向性的大小值，如數量、距離、速度、溫度)
# 1. vector 向量(同時具有大小及方向性，向量更多地被稱作矢量；矢量可以描述許多常見的物理量，如運動學中的位移、速度、加速度，力學中的力、力矩，電磁學中的電流密度、磁矩、電磁波等等。)
# 2. matrix 矩陣
# 3. n-dimensional tensor 多維度張量

# # scalar 純量==============================================================================================
# # import torch
# from torch import tensor
# x = tensor(42.)
# print(x)
# x.dim()         # 維度
# print(x.dim())
# print(2 * x)    # 可以運算
# # 常用的資料格式就是 tensor，但有時為了方便我們查看數值，我們需要將他轉回純量。
# print(x.item()) # pytorch 中 item() 用於只包含一個元素的tensor中提取值，如果不只包含一個元素則使用 tolist()

# # vector 向量==============================================================================================
# # 例如: [-5., 2., 0.]   [身高, 體重, 年齡] 在深度學習中通常指特徵，多個指標，例如:詞向量特徵、某一維度特徵
# import torch
# from torch import tensor
# v = tensor([1.5, -0.5, 3.0])
# print(v)
# print(f'維度: {v.dim()}')
# print(f'維度大小: {v.size()}')

# matrix 矩陣==============================================================================================
# 一般拿來計算矩陣，通常是多維的
import torch
from torch import tensor
m = tensor([[1., 2.],[3., 4.]])
print(m)
print(m.matmul(m))                      # .matmul() 矩陣乘法
print(tensor([1., 0.]).matmul(m))
print(m*m)                              # 會依照同行同列相乘
print(tensor([1., 2.]).matmul(m))