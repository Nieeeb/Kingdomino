import cv2 as cv
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


# Funktion til at finde alle tiles i en gruppe af sammenhængende tiles med samme label
def find_tile_group(tile_index, tiles_dict, visited):
    # Initialiser liste til at gemme tiles i gruppen
    tile_group = []
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
def find_all_tile_groups(tiles_dict):
    visited = set()
    tile_groups = []

    for tile_index in tiles_dict:
        if tile_index not in visited:
            tile_group = find_tile_group(tile_index, tiles_dict, visited)
            tile_groups.append(tile_group)

    return tile_groups

# Funktion til at tælle point i gruppéringerne
def count_points(tiles_dict, crown_dict, all_tile_groups):
    total_points = 0
    unknown_exists = False
    for groups in all_tile_groups:
        crowns_in_group = 0
        for tile in groups:
            if tiles_dict[tile] == 'unknown':
                # Tjek alle tiles for unknowns
                unknown_exists = True

            crowns_in_group += crown_dict[tile]
            if tile == (2, 2) and tiles_dict[tile] == 'home':
                # Tjek midten for home-tilen
                total_points += 10

        total_points += crowns_in_group*len(groups)

    if not unknown_exists:
        total_points += 5

    return total_points

# Denne funktion er en samling af alle de funktioner, der skal bruges til at regne point.
def count_points_in_image(image, classifier):
    df = define_tiles_for_image(classifier, image)
    tiles_dict = create_dict_with_pos_and_label(df)
    crown_dict = create_dict_with_pos_and_crowncount(df)
    all_tile_groups = find_all_tile_groups(tiles_dict)
    total_points = count_points(tiles_dict, crown_dict, all_tile_groups)
    return total_points, tiles_dict, all_tile_groups

# Denne main() bliver kun brugt til at teste scriptet
def main():
    # Usual path extraction
    path = r"King Domino dataset/Full game areas/2.jpg"
    unknown_image = cv.imread(path)

    # initier knn
    knn = KNeighborsClassifier(n_neighbors=3)
    data = load_data()
    X_train, y_train, X_test, y_test = complete_split(data)
    knn.fit(X_train, y_train)

    total_points, tiles_dict, all_tile_groups = count_points_in_image(unknown_image, knn)

    print('Total points in image: ', total_points)

    # Udskriv antallet af tiles i hver gruppe
    for i, tile_group in enumerate(all_tile_groups):
        label = tiles_dict[tile_group[0]]
        print(f"Gruppe {i + 1}: Antal tiles = {len(tile_group)}, Label = {label}")

if __name__ == '__main__':
    main()
