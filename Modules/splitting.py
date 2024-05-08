import cv2 as cv
import os
import numpy as np
import pandas as pd
from imutils.object_detection import non_max_suppression 
import glob
import math

def calculate_color_values(tile):
    rgbTile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
    meanR, meanG, meanB = np.mean(rgbTile, axis=(0,1))
    medR, medG, medB = np.median(rgbTile, axis=(0,1))
    
    hsvTile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    meanH, meanS, meanV = np.mean(hsvTile, axis=(0,1))
    medH, medS, medV = np.median(hsvTile, axis=(0,1))
    
    dict = {'medH': medH,'medS': medS,'medV': medV,'medR': medR,'medG': medG,'medB': medB,'meanR': meanR,'meanG': meanG,'meanB': meanB,'meanH': meanH,'meanS': meanS,'meanV': meanV}
    return dict

def split_image(image):
    tile_centers = calculate_tile_centers()
    templates = create_templates()
    boxes = give_matching_boxes(templates, image)
    box_centers = calculate_box_centers(boxes)
    tiles_with_crowns = find_tiles_with_crowns(tile_centers, box_centers)
    
    size = 500
    tiles_per_side = 5
    cut_off_size = 5
    tile_size = (size // tiles_per_side)
    large_tile_size = (size // tiles_per_side)
    cut_images = []
    for i in range(tiles_per_side):
        for j in range(tiles_per_side):
            cut = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            cut = cut[cut_off_size:tile_size - cut_off_size, cut_off_size:tile_size - cut_off_size]
            cut_save = {'tilePos': (i, j), 'cut_image': cut}
            
            image_colors = calculate_color_values(cut)
            # crowns = {'crowns': count_crowns_in_tile(cut)}
            crown_count = 0
            for tile_index in tiles_with_crowns:
                if tile_index == (i, j):
                    crown_count += 1
            crowns = {'crowns': crown_count}
            final_dict = cut_save | image_colors | crowns
            cut_images.append(final_dict)
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
        # temp_gray = convert_to_grayscale(template)
        # image_gray = convert_to_grayscale(image)
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

def create_templates():
    templates = [cv.imread(file) for file in glob.glob(r"Modules/Templates/*.png")]
    
    rotated = []
    for template in templates:
        rotated += rotate_image(template)
    return rotated

def count_crowns_in_tile(tile):
    templates = create_templates()
    boxes = give_matching_boxes(templates, tile)
    count = give_number_of_crowns(boxes)
    return count

def define_tiles_for_image(classifier, image):
    cut_images = split_image(image)
    df = pd.DataFrame(cut_images)
    X = df.drop(['tilePos', 'cut_image', 'crowns'], axis=1)
    labels = classifier.predict(X)
    df['labels'] = labels
    return df

def create_dict_with_pos_and_label(df):
    output = {}
    for index, row in df.iterrows():
        output = output | {row['tilePos']: row['labels']}
    return output

def create_dict_with_pos_and_crowncount(df):
    output = {}
    for index, row in df.iterrows():
        output = output | {row['tilePos']: row['crowns']}
    return output

def calculate_tile_centers():
    image_size = 500
    tiles_per_side = 5
    tile_side_length = 500//5
    centers = {}
    for y in range(tiles_per_side):
        for x in range(tiles_per_side):
            index = (y, x)
            centerx = (image_size//tiles_per_side)/2 + x*tile_side_length 
            centery = (image_size//tiles_per_side)/2 + y*tile_side_length 
            centerPoint = (centerx, centery)
            # print(centerPoint)
            centers[index] = centerPoint
    return centers
            

def calculate_box_centers(boxes):
    # print(boxes)
    centers = []
    for box in boxes:
        topX = box[0]
        topY = box[1]
        bottomX = box[2]
        bottomY = box[3]
        
        centerX = (bottomX + topX)//2
        centerY = (bottomY + topY)//2
        centerpoint = (centerX, centerY)
        centers.append(centerpoint)
    return centers
    

def calculate_distance(point_A, point_B):
    distance = math.dist(point_A, point_B)
    # print(distance)
    return distance

def find_tiles_with_crowns(tile_centers, box_centers):
    tiles_with_crowns = []
    for box_center in box_centers:
        current_shortest = 999999
        shortest_index = None
        for tile_index, tile_center in tile_centers.items():
            distance = calculate_distance(box_center, tile_center)
            if distance < current_shortest:
                current_shortest = distance
                shortest_index = tile_index
                # print(tile_index)
        tiles_with_crowns.append(shortest_index)
    return tiles_with_crowns

def main():
    #path = os.path.abspath(__file__ + '/../../../') + f'\King Domino dataset\Cropped and perspective corrected boards\\4.jpg'
    #path = os.path.dirname(os.getcwd()) + '\King Domino dataset\Cropped and perspective corrected boards\\1.jpg'
    path = r"King Domino dataset\Cropped and perspective corrected boards\22.jpg"
    image = cv.imread(path)
    #cv.imshow("Board", image)
    cut_images = split_image(image)
    df = pd.DataFrame(cut_images)
    print(df['crowns'])
    tile = cut_images[16]['cut_image']
    
    centers = calculate_tile_centers()
    print(centers)
    
    rotated = create_templates()
    
    boxes = give_matching_boxes(rotated, image)
    box_centers = calculate_box_centers(boxes)
    
    crowns_in_tiles = find_tiles_with_crowns(centers, box_centers)
    print(crowns_in_tiles)
    
    # print(boxes)
    drawn = draw_boxes(image, boxes)
    crowns_found = give_number_of_crowns(boxes)
    print(f"Crowns Found: {crowns_found}")
    # Show the template and the final output 
    cv.imshow("After NMS", drawn) 
    
    cv.waitKey()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()