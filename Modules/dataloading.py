from splitting import *
import pandas as pd
import cv2 as cv
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import ast

def load_image_from_id(imageid):
    path = f"King Domino dataset/Cropped and perspective corrected boards/{imageid}.jpg"
    return cv.imread(path)

def give_specific_tile(tileid, image):
    cut_images = split_image(image)
    tile = cut_images[tileid]['cut_image']
    return tile

def give_tile_by_image_and_tile_id(imageid, tileid):
    image = load_image_from_id(imageid)
    tile = give_specific_tile(tileid, image)
    return tile

def give_tile_by_image_and_position(imageid, tilePosition):
    tileId = 5 + tilePosition[0] * tilePosition[1]
    tile = give_tile_by_image_and_position(imageid, tileId)
    return tile

def convert_hsv_data():
    raw_data = pd.read_csv(r"Modules/NimRod/hsv_training.csv")
    image_id = []
    tiles = []
    for i in range(73):
        image_id.append(i)
        tile_id = []
        for j in range(25):
            tile_id.append(j)
        tiles += tile_id
    raw_data['tileId'] = tiles
    raw_data = raw_data.rename(columns={'tile': 'tilePosition'})   
    raw_data = raw_data.drop(['hsv'], axis=1)
    raw_data.to_csv(r"King Domino dataset/raw.csv", index=False)
    print(raw_data)
    
def attach_hsv_data(raw_data):
    working_data = raw_data.copy()
    h_collected = []
    s_collected = []
    v_collected = []
    for index, row in working_data.iterrows():
        tile = give_tile_by_image_and_tile_id(row['image'], row['tileId'])
        hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        h, s, v = np.median(hsv_tile, axis=(0, 1))
        h_collected.append(h)
        s_collected.append(s)
        v_collected.append(v)
    working_data['medH'] = h_collected
    working_data['medS'] = s_collected
    working_data['medV'] = v_collected

    return working_data

def attach_rgb_data(raw_data):
    working_data = raw_data.copy()
    r_collected = []
    g_collected = []
    b_collected = []
    for index, row in working_data.iterrows():
        tile = give_tile_by_image_and_tile_id(row['image'], row['tileId'])
        b, g, r = np.median(tile, axis=(0, 1))
        r_collected.append(r)
        g_collected.append(g)
        b_collected.append(b)
    working_data['medR'] = r_collected
    working_data['medG'] = g_collected
    working_data['medB'] = b_collected
    
    return working_data

def attach_means(raw_data):
    working_data = raw_data.copy()
    r_collected = []
    g_collected = []
    b_collected = []
    h_collected = []
    s_collected = []
    v_collected = []
    for index, row in working_data.iterrows():
        tile = give_tile_by_image_and_tile_id(row['image'], row['tileId'])
        b, g, r = np.mean(tile, axis=(0, 1))
        r_collected.append(r)
        g_collected.append(g)
        b_collected.append(b)
        hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        h, s, v = np.mean(hsv_tile, axis=(0, 1))
        h_collected.append(h)
        s_collected.append(s)
        v_collected.append(v)
    working_data['meanR'] = r_collected
    working_data['meanG'] = g_collected
    working_data['meanB'] = b_collected
    working_data['meanH'] = h_collected
    working_data['meanS'] = s_collected
    working_data['meanV'] = v_collected
    
    return working_data

def pick_training_ids(percentage, imagecount):
    ids_to_keep = [i for i in range(imagecount)]
    random.Random(69).shuffle(ids_to_keep)
    count_to_remove = round(imagecount * percentage)
    ids_to_remove = ids_to_keep[:count_to_remove]
    
    return ids_to_remove

def split_data(raw_data):
    working_data = raw_data.copy()
    ids_to_remove = pick_training_ids(0.2, 73)
    train_data = working_data[~working_data['image'].isin(ids_to_remove)]
    test_data = working_data[working_data['image'].isin(ids_to_remove)]
    return train_data, test_data

def give_x_and_y(raw_data):
    working_data = raw_data.copy()
    labelColumn = 'label'
    discardColumns = ['image', 'label', 'tileId', 'tilePosition']
    
    y = working_data[labelColumn]
    x = working_data.drop(discardColumns, axis=1)
    
    return x, y

def complete_split(raw_data, giveValidationSet=False):
    train_data, test_data = split_data(raw_data)
    x_test, y_test = give_x_and_y(test_data)
    if giveValidationSet:
        x_train, y_train = give_x_and_y(train_data)
        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, random_state=42, test_size=0.2, shuffle=True)
        return x_train, y_train, x_validate, y_validate, x_test, y_test
    else:
        x_train, y_train = give_x_and_y(test_data)
        return x_train, y_train, x_test, y_test

def attach_then_save_data():
    raw_data = pd.read_csv(r"King Domino dataset/raw.csv")
    attached = attach_hsv_data(raw_data)
    attached = attach_rgb_data(attached)
    attached = attach_means(attached)
    attached.to_csv(r"King Domino dataset/attached.csv", index=False)

def load_data():
    path = os.path.abspath(__file__ + '/../../../') + f'\Kingdomino\King Domino dataset/attached.csv'
    raw_data = pd.read_csv(path)
    return raw_data

def main():
    #convert_hsv_data()
    #attach_then_save_data()
    data = load_data()
    #print(data)
    
    x_train, y_train, x_test, y_test = complete_split(data)
    
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn.fit(x_train, y_train)
    
    path = r"King Domino dataset/Full game areas/2.jpg"
    image = cv.imread(path)
    print(define_tiles_for_image(knn, image))
    
    #predictions = knn.predict(x_test)
    #print(classification_report(y_pred=predictions, y_true=y_test))
    
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
