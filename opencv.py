
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('res/real1.png', 0)
edges = cv2.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')  # 1 row, 2 columns, position 2
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# img2 = cv2.imread('res/real1.png')
# result = cv2.pencilSketch(img2)
# plt.imshow(img2)
# plt.title("Pencil Sketch")

plt.show()
