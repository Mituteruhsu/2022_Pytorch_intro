# P36. 圖像基本操作
import cv2      # OpenCV 默認的格式為 BGR
import matplotlib.pyplot as plt
import numpy as np
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# vc = cv2.VideoCapture('test.mp4')
# if vc.isOpened():
#     ocpn, frame = vc.read()
# else:
#     open = False

# while open:
#     rel, frame = vc.read()
#     if frame is None:
#         break
#     if rel == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result', gray)
#         if cv2.waitKey(10) & 0xFF == 27:
#             break
# vc.release()
# cv2.destroyAllWindows()

# ROI region of interest========================
img = cv2.imread('cat.jpg')
cat = img[200:350, 0:200]       # [x 軸高度, y 軸寬度]
# cv_show('cat', cat)

b, g, r = cv2.split(img)
# print(r)
# print(r.shape)

# 顏色通道提取====================================
b, g, r = cv2.split(img)
# print(b)
# print(b.dtype)
# print(b.shape)
# 顏色通道組合===================================
img = cv2.merge((b, g, r))
# print(img.shape)

# 只保留 BGR 中的指定通道=======================
# R 通道
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
# cv_show('red', cur_img)

# G 通道
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 2] = 0
# cv_show('green', cur_img)

# B 通道
cur_img = img.copy()
cur_img[:, :, 1] = 0
cur_img[:, :, 2] = 0
# cv_show('blue', cur_img)

# 邊界填充===========================================================================================
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)

import matplotlib.pyplot as plt
# plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
# plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')     # 複製最邊的邊界延長
# plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')         # 從最邊直接向外反射複製 例如: hgfedcba[abcdefgh]hgfedcba
# plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT101')   # 以邊為軸向外倒轉圖複製 例如: hgfedcb[abcdefgh]gfedcba
# plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')               # 外包裝法 例如: abcdefgh[abcdefgh]abcdefgh
# plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')       # 常量法, 常數值填充

# plt.show()

# 數值計算=======================================================================
img_cat = cv2.imread('cat.jpg')
img_dog = cv2.imread('dog.jpg')
img_cat2 = img_cat + 70
# print(img_cat[:5, :, 0])    # 只列出前 5 行
# print(img_cat2[:5, :, 0])   # 所有數值都加 70
# print((img_cat+img_cat2)[:5, :, 0]) # 當相加數值大於255, 會以相加數值 除 256 的餘數顯示 164+94=258 258/256= 1 % 2
# print(cv2.add(img_cat, img_cat2)[:5, :, 0]) # numpy 中會取最大值255, 不另做計算

# 圖像融合=======================================
# print(img_cat + img_dog)    # 因為兩張圖的大小不同會報錯 (414,500,3) (429,499,3)
# print(f'Cat_size: {img_cat.shape}',f'\nDog_size: {img_dog.shape}')    # 確認shape
img_dog = cv2.resize(img_dog, (500, 414))   # 改變大小
# print(img_dog.shape)
#                       α   ,   x1,     β  ,  x2, b
res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)    # R = α(x1) + β(x2) + b
# print(plt.imshow(res))  # 兩個修改後的大小
# plt.imshow(res)
# plt.show()

res = cv2.resize(img, (0, 0), fx=3, fy=1)   # fx , fy, 指定倍數
# print(plt.imshow(res))
# plt.show()

res = cv2.resize(img, (0, 0), fx=1, fy=3)   # fx , fy, 指定倍數
print(plt.imshow(res))
plt.show()