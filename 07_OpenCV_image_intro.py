# P36. 圖像基本操作
import cv2      # OpenCV 默認的格式為 BGR
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('cat.jpg')
# print(img)  # [h, w, c] 顯示所有維度

# cv2.imshow('image', img)
# cv2.waitKey(0)      # time of wait '0' infinity
# cv2.destroyAllWindows()
# ↓↓↓↓↓↓↓↓↓↓↓↓ def ↓↓↓↓↓↓↓↓↓↓↓↓
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cv_show('cat', img)
# print(img.shape)
img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)   # 轉換為灰白圖
# print(img)
# print(img.shape)
# cv_show('cat_gray', img)

# 影像保存
# cv2.imwrite('cat.png', img)

# 圖像的 type = numpy.ndarray
# print(type(img))

# 像數點總個數 = 414 * 500 = 207000
print(img.size)

# 查看數據類型
print(img.dtype)