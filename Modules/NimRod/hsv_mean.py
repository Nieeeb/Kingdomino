import numpy as np
import pandas as pd
import os
import cv2 as cv
from Kingdomino.Modules.TileSplitter import get_tiles

df = pd.read_csv('hsv_training.csv')
mean_rgb_hsv = []
label = df['label']
print(len(label))
j = 0
for i in range(1, 74):
    path = os.path.abspath(
        __file__ + '/../../../') + f'\King Domino dataset\Cropped and perspective corrected boards\\{i}.jpg'
    image = cv.imread(path)

    tiles = get_tiles(image)
    for x, row in enumerate(tiles):
        for y, tile in enumerate(row):
            hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
            rgb_tile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
            h, s, v = np.mean(hsv_tile, axis=(0, 1))
            r, g, b = np.mean(rgb_tile, axis=(0, 1))

            mean_rgb_hsv.append(
                {"image": i, "tile": (x, y), "label": label[j], "r": r, "g": g, "b": b, "h": h, "s": s, "v": v})
            j += 1
            print(j)
df1 = pd.DataFrame(mean_rgb_hsv)
df1.to_csv('mean_rgb_hsv.csv', index=False)
