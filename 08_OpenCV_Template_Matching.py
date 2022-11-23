import cv2
import matplotlib.pyplot as plt
import numpy as np

# 模板匹配(Template Matching)是一種在較大圖像中搜索和查找範本圖像位置的方法。

img = cv2.imread('lena.jpg', 0)
template = cv2. imread('face.jpg', 0)
h, w = template.shape[:2]
print(img.shape)
print(template.shape)
# 模板匹配和卷積原理相似, 模板在原圖像上從原點開始滑動, 計算模板與(圖像被模板覆蓋的地方)之差別程度, 這個差別程度的計算方式在OpenCV中有 6 種
# TM_SQDIFF:(平方差匹配)    計算平方不同, 計算出來的值越小, 越相關。
# TM_CCORR:(相關匹配)       計算相關性, 計算出來的值越大, 越相關。
# TM_CCOEFF:(相關匹配)      計算相關係數, 計算出來的值越大, 越相關。
# TM_SQDIFF_NORMED:(標準相關匹配)   計算標準化平方不同, 計算出來得值越接近 0, 越相關。
# TM_CCORR_NORMED:(標準相關匹配)    計算標準化後的相關性, 計算出來的值越接近 1, 越相關。
# TM_CCOEFF_NORMED:(標準相關匹配)   計算標準化後的相關係數, 計算出來的值越接近 1, 越相關。
# OpenCV 會從左到右, 從上到下一一去匹配。

# 每次計算的結果會放入一個矩陣中, 作為結果輸出。
# 假設原圖為 A x B 大小, 而模板是 a x b 大小, 輸出結果的矩陣為 (A - a + 1) X (B - b + 1)
# (A-a+1, B-b+1)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
# print(res.shape)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# print(f'min_val: {min_val}')
# print(f'max_val: {max_val}')
# print(f'min_loc: {min_loc}')    # 最小值:最左上角的座標
# print(f'max_val: {max_loc}')    # 最大值:最右下角的座標

# 6 種方式的比較
for meth in methods:
    img2 = img.copy()

    # 匹配方法的真值
    method = eval(meth)
    print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果是平方差匹配 TM_SQDIFF 或 TM_SQDIFF_NORMED 標準化平方差匹配, 取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 繪圖(方形圖)
    # cv2.rectangle(img, top_left, bottom_right, 255, 2)

    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.xticks([]), plt.yticks([])      # 隱藏坐標軸
    # plt.subplot(122), plt.imshow(img2, cmap='gray')
    # plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()

# 匹配多個對象===========================================
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2. imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配大於 80% 的座標
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)