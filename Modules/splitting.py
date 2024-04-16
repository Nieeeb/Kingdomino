import cv2 as cv
import os
import numpy as np
from imutils.object_detection import non_max_suppression 
import glob

def split_image(image):
    size = 500
    tiles_per_side = 5
    cut_off_size = 0
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

def generate_surf_features(image):
    working_image = image.copy()
    working_image = convert_to_grayscale(working_image)
    sift = cv.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(working_image, None)
    return keypoints_sift, descriptors_sift

def display_keypoints_on_image(image, keypoints):
    img_kp = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow(f"{image.size}", img_kp)

def give_matching_boxes(templates, image):
    boxes = []
    for template in templates:
        temp_gray = convert_to_grayscale(template)
        image_gray = convert_to_grayscale(image)
        result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        
        threshold = 0.7
        
        (y_points, x_points) = np.where(result >= threshold)
        
        h, w, _ = template.shape
        for x, y in zip(x_points, y_points):
            boxes.append((x,y, x + w, y + h))
            
    boxes = non_max_suppression(np.array(boxes))
    
    return boxes

def draw_boxes(image, boxes):
    result = image.copy()
    for (x1, y1, x2, y2) in boxes: 
        # draw the bounding box on the image 
        cv.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return result

def rotate_image(image):
    rotated_images = [image, None, None, None]
    
    for i in range(3):
        rotated_images[i+1] = cv.rotate(rotated_images[i], cv.ROTATE_90_CLOCKWISE)
    
    
    return rotated_images

def give_number_of_crowns(boxes):
    return len(boxes)

def main():
    #path = os.path.abspath(__file__ + '/../../../') + f'\King Domino dataset\Cropped and perspective corrected boards\\4.jpg'
    #path = os.path.dirname(os.getcwd()) + '\King Domino dataset\Cropped and perspective corrected boards\\1.jpg'
    path = r"King Domino dataset/Cropped and perspective corrected boards/32.jpg"
    image = cv.imread(path)
    #cv.imshow("Board", image)
    cut_images = split_image(image)
    tile = cut_images[1]['cut_image']
    
    templates = [cv.imread(file) for file in glob.glob(r"Modules/Templates/*.png")]
    
    rotated = []
    for template in templates:
        rotated += rotate_image(template)
        
    boxes = give_matching_boxes(rotated, tile)
    
    drawn = draw_boxes(tile, boxes)
    crowns_found = give_number_of_crowns(boxes)
    print(f"Crowns Found: {crowns_found}")
    # Show the template and the final output 
    cv.imshow("After NMS", drawn) 
    
    cv.waitKey()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()