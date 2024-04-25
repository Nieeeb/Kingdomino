import os
import cv2 as cv
import ast
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from dataloading import *




# Funktion til at finde nabo-tiles baseret på tile index
def find_neighbor_tiles(tile_index):
    row_index, col_index = tile_index
    neighbors = []

    # Tjek op, ned, venstre og højre naboer
    # Op
    if row_index > 0:
        neighbors.append((row_index - 1, col_index))
    # Ned
    if row_index < 4:
        neighbors.append((row_index + 1, col_index))
    # Venstre
    if col_index > 0:
        neighbors.append((row_index, col_index - 1))
    # Højre
    if col_index < 4:
        neighbors.append((row_index, col_index + 1))

    return neighbors


# Funktion til at tælle antallet af naboer af samme slags
def count_same_type_neighbors(tile_index, tiles_dict):
    visited = set()
    tile_group = find_tile_group(tile_index, tiles_dict, visited)
    return len(tile_group)


# Funktion til at finde alle tiles i en gruppe af sammenhængende tiles med samme label
def find_tile_group(tile_index, tiles_dict, visited):
    # Initialiser en liste til at gemme tiles i gruppen
    tile_group = []

    # Stack til DFS
    stack = [tile_index]

    while stack:
        current_tile = stack.pop()
        tile_group.append(current_tile)
        visited.add(current_tile)

        # Find naboer af den aktuelle tile
        neighbors = find_neighbor_tiles(current_tile)

        # Tilføj naboer til stacken, hvis de har samme label og ikke er besøgte
        for neighbor in neighbors:
            if neighbor in tiles_dict and tiles_dict[neighbor] == tiles_dict[tile_index] and neighbor not in visited:
                stack.append(neighbor)
                visited.add(neighbor)  # Marker nabo-tile som besøgt

    return tile_group


# Funktion til at finde alle forskellige grupper af sammenhængende tiles med samme label
def find_all_tile_groups(tiles_dict, crown_dict):
    visited = set()
    tile_groups = []

    for tile_index in tiles_dict:
        if tile_index not in visited:
            tile_group = find_tile_group(tile_index, tiles_dict, visited)
            tile_groups.append(tile_group)

    return tile_groups


# Læs csv'en med det labelt data
# df = pd.read_csv('hsv_training.csv')
#
# # extraction som np arrays
# labels = np.array(df['label'].values)
# hsv_values_strs = np.array(df['hsv'].values)
# # tuplerne i datasættet er åbenbart string lmao
# hsv_values = [ast.literal_eval(medians) for medians in hsv_values_strs]

# Usual path extraction
path = os.path.abspath(__file__ + '/../../../') + f'\King Domino dataset\Full game areas\\2.jpg'
unknown_image = cv.imread(path)
# tiles = get_tiles(unknown_image)

# Brugt til at generere unlabeled HSV datapunkter til knn'en
# unknown_hsv_values = []
# tiles_indeces = []
# for x, row in enumerate(tiles):
#     for y, tile in enumerate(row):
#         hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
#         h, s, v = np.median(hsv_tile, axis=(0, 1))
#         unknown_hsv_values.append([h, s, v])
#         tiles_indeces.append((x, y))
# print(tiles_indeces)

# initier knn
knn = KNeighborsClassifier(n_neighbors=3)
data = load_data()
X_train, y_train, X_test, y_test = complete_split(data)
knn.fit(X_train, y_train)

df = define_tiles_for_image(knn, unknown_image)
tiles_dict = create_dict_with_pos_and_label(df)
crown_dict = create_dict_with_pos_and_crowncount(df)
print("crown dict", crown_dict)
# unknown_X_test = np.array(unknown_hsv_values)
# y_pred = knn.predict(unknown_X_test)

# print("Forudsagte labels:", y_pred)

# for x, row in enumerate(tiles_indeces):
#     tiles_dict[row] = y_pred[x]
# print(tiles_dict)

# Find alle forskellige grupper af sammenhængende tiles med samme label
all_tile_groups = find_all_tile_groups(tiles_dict, crown_dict)

# Udskriv antallet af tiles i hver gruppe
for i, tile_group in enumerate(all_tile_groups):
    label = tiles_dict[tile_group[0]]
    print(f"Gruppe {i + 1}: Antal tiles = {len(tile_group)}, Label = {label}")
