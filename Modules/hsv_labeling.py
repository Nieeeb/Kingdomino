import cv2 as cv
import numpy as np
import os
import pandas as pd
from TileSplitter import get_tiles

#Indlæser alle billederne i Cropped and perspective corrected boards-mappen en af gangen
#Og splitter dem op i tiles, spørger efter hvilket label tilen skal have
#og dumber det hele i en csv fil, når man har labelt samtlige 1825 tiles:))))))))))))))))))))
hsv_labels = []
for i in range(1, 74):
    path = f'\King Domino dataset\Cropped and perspective corrected boards\\{i}.jpg'
    image = cv.imread(path)
    print(path)
    tiles = get_tiles(image)


    for x, row in enumerate(tiles):
        for y, tile in enumerate(row):
            winname = f"tile {x, y}"
            cv.imshow(winname, tile)
            cv.waitKey(1)
            hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
            b, g, r = np.median(tile, axis=(0, 1))
            h, s, v = np.median(hsv_tile, axis=(0, 1))
            label = input("Enter label: ")
            hsv_labels.append({"image": i, "tile": (x, y), "label": label, "hsv": (h, s, v)})
            cv.destroyWindow(winname)

df = pd.DataFrame(hsv_labels)
df.to_csv("hsv_training.csv", index=False)

cv.destroyAllWindows()
