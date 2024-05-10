import cv2 as cv
import numpy as np
import pandas as pd
from imutils.object_detection import non_max_suppression 
import glob
import math

# Funktion der udregner HSV og RGB værdier for et givent tile
def calculate_color_values(tile):
    rgbTile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
    meanR, meanG, meanB = np.mean(rgbTile, axis=(0,1))
    medR, medG, medB = np.median(rgbTile, axis=(0,1))
    
    hsvTile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    meanH, meanS, meanV = np.mean(hsvTile, axis=(0,1))
    medH, medS, medV = np.median(hsvTile, axis=(0,1))
    
    dict = {
        'medH': medH,
        'medS': medS,
        'medV': medV,
        'medR': medR,
        'medG': medG,
        'medB': medB,
        'meanR': meanR,
        'meanG': meanG,
        'meanB': meanB,
        'meanH': meanH,
        'meanS': meanS,
        'meanV': meanV
    }
    return dict

# Funktion der finder matchende templates
# Giver liste af kasser der passer på fundne kroner
def give_matching_boxes(templates, image):
    boxes = []
    for template in templates:
        result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        
        threshold = 0.7
        
        (y_points, x_points) = np.where(result >= threshold)
        
        h, w, _ = template.shape
        for x, y in zip(x_points, y_points):
            boxes.append((x,y, x + w, y + h))
    
    # Filtrerer overlappende kasser fra
    boxes = non_max_suppression(np.array(boxes))
    
    return boxes

# Hjælpefunktion til at vise template matches på skærmen
def draw_boxes(image, boxes):
    result = image.copy()
    for (x1, y1, x2, y2) in boxes: 
        # draw the bounding box on the image 
        cv.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return result

# Funktion der modtager et billede og returnerer 4 forskellige rotationer af det
# Brugt til at skabe forskellige versioner af templates
def rotate_image(image):
    rotated_images = [image, None, None, None]
    
    for i in range(3):
        rotated_images[i+1] = cv.rotate(rotated_images[i], cv.ROTATE_90_CLOCKWISE)
    
    return rotated_images

# Funktion der læser alle templates i templates mappen
# Laver roterede versioner af templates
def create_templates():
    templates = [cv.imread(file) for file in glob.glob(r"Modules/Templates/*.png")]
    
    rotated = []
    for template in templates:
        rotated += rotate_image(template)
    return rotated

# Funktion der modtager billede af board og en classifier for at label tiles
# Classifier forventes at være trænet
def define_tiles_for_image(classifier, image):
    # Splitter billede
    cut_images = split_image(image)
    # Omdanner liste af dictionaries til dataframe
    df = pd.DataFrame(cut_images)
    # Udtager kun farveværdier
    X = df.drop(['tilePos', 'cut_image', 'crowns'], axis=1)
    # Laver predictions
    labels = classifier.predict(X)
    df['labels'] = labels
    return df

# Funktion der skaber et dictionary med tileposition som key og label som value
# Brugt til at gruppere tiles sammen
def create_dict_with_pos_and_label(df):
    output = {}
    for index, row in df.iterrows():
        output = output | {row['tilePos']: row['labels']}
    return output

# Funktion der skaber et dictionary med tileposition som key og crown antal som value
# Brugt til at tælle kroner i grupper
def create_dict_with_pos_and_crowncount(df):
    output = {}
    for index, row in df.iterrows():
        output = output | {row['tilePos']: row['crowns']}
    return output

# Funktioner der udregner midtpunkter for et billede
# Der antages at alle billeder er 500x500 pixels og det er et 5x5 grid af tiles
# Returnerer et dictionary med tile position som key og koordinat som value
# Brugt til at definere hvilke tiles kroner hører til
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

            centers[index] = centerPoint
    return centers
            
# Funktion der modtager kasser fra template matching og udregner midtpunkter
# Input består af top venstre hjørne og bund højre hjørne for hver kasse
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
    
# Funktion der udregner euklidiskt afstand mellem to punkter
def calculate_distance(point_A, point_B):
    distance = math.dist(point_A, point_B)
    return distance

# Funktioner der matcher template kasser med tætteste tile midtpunkt
# KNN med K = 1
# Det tætteste midpunkt sættes som at være en crowns tile
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

# Samlig af alle crown tælling funktioner
# Modtager 1 billede og giver en liste med tætteste index for hver krone
def count_crowns(image):
    # Udregner midpunkter til hvert tile
    tile_centers = calculate_tile_centers()
    # Klagør templates
    templates = create_templates()
    # Template matcher på billedet med templates
    boxes = give_matching_boxes(templates, image)
    # Finder midtpunkterne af de matchede kroner
    box_centers = calculate_box_centers(boxes)
    # Finder tætteste til midtpunkt til hver kasse
    tiles_with_crowns = find_tiles_with_crowns(tile_centers, box_centers)
    return tiles_with_crowns

# Funktion der modtager et billede og giver en samling a tiles
# Hver tile får udregnet HSV og RGB værdier, samt antal crowns fundet
def split_image(image):
    # Finder hvilke tile index er er crowns i.
    # Et givent tile index kan forekomme flere gange hvis der er mere end 1 crown
    tiles_with_crowns = count_crowns(image)
    
    size = 500
    tiles_per_side = 5
    cut_off_size = 5
    tile_size = (size // tiles_per_side)

    # Udskærer 25 tiles fra original billedet
    cut_images = []
    for i in range(tiles_per_side):
        for j in range(tiles_per_side):
            cut = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            cut = cut[cut_off_size:tile_size - cut_off_size, cut_off_size:tile_size - cut_off_size]
            cut_save = {'tilePos': (i, j), 'cut_image': cut}
            # Udregner farveværdier
            image_colors = calculate_color_values(cut)

            # Matcher tile index for nuværende med de tile index med kroner i og tæller antallet af crowns
            crown_count = 0
            for tile_index in tiles_with_crowns:
                if tile_index == (i, j):
                    crown_count += 1
            crowns = {'crowns': crown_count}
            
            # Samler alt til 1 stort dictionary
            final_dict = cut_save | image_colors | crowns
            cut_images.append(final_dict)
    return cut_images

def main():
    path = r"King Domino dataset\Cropped and perspective corrected boards\1.jpg"
    image = cv.imread(path)
    
    centers = calculate_tile_centers()
    
    rotated = create_templates()
    
    boxes = give_matching_boxes(rotated, image)
    box_centers = calculate_box_centers(boxes)
    
    crowns_with_tiles = find_tiles_with_crowns(centers, box_centers)
    print(crowns_with_tiles)
    
    drawn = draw_boxes(image, boxes)
    # Show the template and the final output 
    cv.imshow("After NMS", drawn) 
    
    path = r"King Domino dataset\Cropped and perspective corrected boards\1.jpg"
    image = cv.imread(path)
    
    path = r"Modules/Templates/templateSwamp.png"
    template = cv.imread(path)
    
    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    cv.imshow("Template Matching", result)
    
    cv.waitKey()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()