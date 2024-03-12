import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y * 100:(y + 1) * 100, x * 100:(x + 1) * 100])
    return tiles

path = os.path.dirname(os.getcwd()) + '\King Domino dataset\Cropped and perspective corrected boards\\1.jpg'
print(f"hah {path}")

#C:\Users\willi\Documents\DAKI Mini Projects\DUAS\Kingdomino\King Domino dataset\Cropped and perspective corrected boards\1.jpg
image = cv.imread(path)
image_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
cv.imshow("Raw", image)

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

tiles = get_tiles(image_hsv)



color = ('b','g','r')

histr = cv.calcHist(tiles[0][0],[1],None,[256],[0,256])
print(histr)
plt.plot(histr,color=color[1])
plt.xlim([0,256])
plt.show()

cv.waitKey()
cv.destroyAllWindows()