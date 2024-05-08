from dataloading import *
from model import *
import pandas as pd
from group_tiles_labels import *

def pre_counted_boards():
    training_ids = [27, 30, 46, 48, 47, 3, 5, 22, 70, 43, 14, 50, 11, 38, 62]
    
    training_images = [
        {'imageID': 27, 'human counted points': 82},
        {'imageID': 30, 'human counted points': 63},
        {'imageID': 46, 'human counted points': 48},
        {'imageID': 48, 'human counted points': 57},
        {'imageID': 47, 'human counted points': 81},
        {'imageID': 3, 'human counted points': 67},
        {'imageID': 5, 'human counted points': 51},
        {'imageID': 22, 'human counted points': 70},
        {'imageID': 70, 'human counted points': 104},
        {'imageID': 43, 'human counted points': 81},
        {'imageID': 14, 'human counted points': 48},
        {'imageID': 50, 'human counted points': 49},
        {'imageID': 11, 'human counted points': 49},
        {'imageID': 38, 'human counted points': 41},
        {'imageID': 62, 'human counted points': 41}
    ]
    
    training_df = pd.DataFrame(training_images)
    return training_df

# Funktion der modtager manuelt optalte point og udregner antal point ud fra funktionerne
# Udregner hvor mange point der bliver talt forkert
def count(boards, model):
    boards_copy = boards.copy()
    
    errors = []
    points = []
    # Kører igennem alle billederne
    for index, imageSeries in boards.iterrows():
        image = load_image_from_id(imageSeries['imageID'])
        points_in_image, _, _, _ = count_points_in_image(image, model)
        points.append(points_in_image)
        
        # Udregner antal point talt forkert
        error = abs(imageSeries['human counted points'] - points_in_image)
        errors.append(error)
        
    boards_copy['machine counted points'] = points
    boards_copy['error'] = errors
    return boards_copy

def main():
    # Indlæser trænede RandomForest model
    # Hvis model filen ikke kan findes, trænes en ny på trænings data
    model = load_trained_model()
    
    # Indlæser test sæt af boards hvor points er talt op manuelt
    boards = pre_counted_boards()
    
    # Tæller point i test sæt
    boards = count(boards, model)
    
    # Udskriver resultater
    print(boards)
    
    # Finder hvor der er fejl:
    print("Images with errors:")
    errors = boards[boards['error'] > 0]
    print(errors)
    print(f"Number of Errors: {len(errors)} out of {len(boards)}")

if __name__ == "__main__":
    main()