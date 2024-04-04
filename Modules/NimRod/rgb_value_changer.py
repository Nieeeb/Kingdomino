import pandas as pd
import cv2 as cv
import os
import numpy as np

def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tile = image[y * 100:(y + 1) * 100, x * 100:(x + 1) * 100]
            tiles[-1].append(tile)
    return tiles

hsv_csv = pd.read_csv('hsv_training.csv')
hsv_csv_copy = hsv_csv.copy()
hsv_csv_copy.rename(columns={'hsv': 'rgb'}, inplace=True)

rgb_values = []
for i in range(1, 74):
    path = os.path.abspath(__file__+'/../../../') + f'\King Domino dataset\Cropped and perspective corrected boards\\{i}.jpg'
    image = cv.imread(path)

    # cv.imshow(f"image {i}", image)
    tiles = get_tiles(image)
    for x, row in enumerate(tiles):
        for y, tile in enumerate(row):
            rgb_tile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
            r, g, b = np.median(rgb_tile, axis=(0, 1))
            print(f"image {i}", f"tile {x, y}", (r, g, b))

            rgb_values.append((r, g, b))

hsv_csv_copy['rgb'] = rgb_values
print(hsv_csv_copy.columns[0])
hsv_csv_copy.to_csv('rgb_training.csv')
