import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from Modules.TileSplitter import get_tiles

path = os.path.dirname(os.getcwd()) + '\King Domino dataset\Cropped and perspective corrected boards\\1.jpg'
print(f"hah {path}")

#C:\Users\willi\Documents\DAKI Mini Projects\DUAS\Kingdomino\King Domino dataset\Cropped and perspective corrected boards\1.jpg
image = cv.imread(path)
image_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
cv.imshow("Raw", image)

image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

tiles = get_tiles(image)
#print(tiles[0])
print(image.dtype)
for x, row in enumerate(tiles):
    for y, tile in enumerate(row):
        cv.imshow(f"tile {x, y}", tile)
        hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        b, g, r = np.median(tile, axis=(0, 1))
        h, s, v = np.median(hsv_tile, axis=(0, 1))
        print(f"tile {x, y} BGR medians: {b, g, r}")
        print(f"tile {x, y} HSV medians: {h, s, v}")


color = ('b','g','r')

histr = cv.calcHist(tiles[0][0],[1],None,[256],[0,256])
#print(histr)
plt.plot(histr,color=color[1])
plt.xlim([0,256])
plt.show()

cv.waitKey()
cv.destroyAllWindows()