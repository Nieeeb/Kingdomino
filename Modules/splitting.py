import cv2 as cv
import os

def split_image(image):
    size = 500
    tiles_per_side = 5
    cut_off_size = 5
    tile_size = size // tiles_per_side
    cut_images = []
    for i in range(tiles_per_side):
        for j in range(tiles_per_side):
            cut = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            cut = cut[cut_off_size:tile_size - cut_off_size, cut_off_size:tile_size - cut_off_size]
            cut_save = {'position': (i, j), 'cut_image': cut}
            cut_images.append(cut_save)
    return cut_images

def display_cuts(cut_images):
    for i in range(len(cut_images)):
        cv.imshow(f"{i}", cut_images[i]['cut_image'])

def convert_to_grayscale(image):
    working_image = image.copy()
    gray = cv.cvtColor(working_image, cv.COLOR_BGR2GRAY) 
    return gray

def binary_threshold_image(image):
    working_image = image.copy()
    discard, binary = cv.threshold(working_image, 127, 255, cv.THRESH_BINARY)
    return binary

def median_filter(image):
    working_image = image.copy()
    blurred = cv.medianBlur(working_image, 5)
    return blurred

path = os.path.dirname(os.getcwd()) + '\Kingdomino\King Domino dataset\Cropped and perspective corrected boards\\1.jpg'

image = cv.imread(path)
cv.imshow("Board", image)

cut_images = split_image(image)
raw = cut_images[18]['cut_image']
cv.imshow("Raw", raw)
gray = convert_to_grayscale(raw)
cv.imshow("Converted", gray)
binary = binary_threshold_image(gray)
print(type(binary))
cv.imshow("Binary", binary)
median = median_filter(binary)
cv.imshow("Median", median)

cv.waitKey()
cv.destroyAllWindows()