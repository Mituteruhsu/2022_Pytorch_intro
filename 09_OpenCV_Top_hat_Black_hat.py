import cv2
import matplotlib.pyplot as plt
import numpy as np
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 顯示原圖
img = cv2.imread('dige.png')
# cv_show('dige', img)

# # EROSION 腐蝕操作================================================
# kernel = np.ones((5, 5), np.uint8)      # 數值越大腐蝕效果越大 5, 5
# erosion = cv2.erode(img, kernel, iterations=1)
# # cv_show('erosion', erosion)

# pie = cv2.imread('pie.png')
# # cv_show('pie', pie)

# kernel = np.ones((30, 30), np.uint8)    # # 數值越大腐蝕效果越大 30, 30
# erosion_1 = cv2.erode(pie, kernel, iterations=1)
# erosion_2 = cv2.erode(pie, kernel, iterations=2)
# erosion_3 = cv2.erode(pie, kernel, iterations=3)
# res = np.hstack((erosion_1, erosion_2, erosion_3))
# # cv_show('res', res)

# Dilation 膨脹操作=================================================
kernel = np.ones((5, 5), np.uint8)      # 數值越大效果越大 5, 5
dige_erosion = cv2.erode(img, kernel, iterations=1)
# cv_show('EROSION', dige_erosion)

kernel = np.ones((5, 5), np.uint8)      # 數值越大效果越大 5, 5
dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)
# cv_show('DILATION', dige_dilate)

pie = cv2.imread('pie.png')

kernel = np.ones((30, 30), np.uint8)    # # 數值越大效果越大 30, 30
erosion_1 = cv2.dilate(pie, kernel, iterations=1)
erosion_2 = cv2.dilate(pie, kernel, iterations=2)
erosion_3 = cv2.dilate(pie, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
# cv_show('RES', res)

# 開運算 與 閉運算  ("Opening operation" & "Closing operation")=============================
img = cv2.imread('dige.png')

# Opening operation 是 先腐蝕 再膨脹----------------------------------
kernel = np.ones((5, 5), np.uint8)      # 數值越大效果越大 5, 5
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv_show('opening', opening)

# Closing operation 是 先膨脹 再腐蝕----------------------------------
kernel = np.ones((5, 5), np.uint8)      # 數值越大效果越大 5, 5
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv_show('closing', closing)

# 梯度運算, 形態學梯度（Morphological Gradient）==========================================
# 針對pie.png   梯度 = 膨脹 - 腐蝕
pie = cv2.imread('pie.png')
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)
res = np.hstack((dilate, erosion))
# cv_show('res', res)

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
# cv_show('gradient', gradient)

# 頂帽與黑帽    ("Top hat" & "Black hat")
# 頂帽 = 原始輸入-開運算結果    (提取出刺來)
# 黑帽 = 閉運算-原始輸入        (只留下一個大致輪廓)

# Top Hat 頂帽 = 原始輸入-開運算結果    (提取出刺來)
img = cv2.imread('dige.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Black Hat 黑帽 = 閉運算-原始輸入  (只留下一個大致輪廓)
img = cv2.imread('dige.png')
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
