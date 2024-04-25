from splitting import *
import cv2 as cv

image = cv.imread(r"King Domino dataset/Full game areas/2.jpg")
crowns = count_crowns_in_tile(image)
print(crowns)