import numpy as np
import os
import cv2 as cv
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk


# Andreas' tilesplitter funktion
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tile = image[y * 100:(y + 1) * 100, x * 100:(x + 1) * 100]
            tiles[-1].append(tile)
    return tiles


def display_image_with_buttons(image):
    # Opret et vindue
    window = tk.Tk()
    window.title("Label Billede")
    window.geometry("200x320+300+200")

    # Indlæs billedet
    photo = Image.fromarray(image)
    photo = ImageTk.PhotoImage(photo)

    # Vis billedet
    label_image = tk.Label(window, image=photo)
    label_image.pack()

    # Liste over gyldige labels
    valid_labels = ["field", "forest", "lake", "grassland", "swamp", "mine", "home", "unknown"]

    # Variabel til at gemme det valgte label
    selected_label = tk.StringVar()

    # Opret knapper for hver label
    for label in valid_labels:
        button = tk.Button(window, text=label, command=lambda l=label: on_button_click(l))
        button.pack()

    def on_button_click(label):
        selected_label.set(label)
        window.destroy()  # Luk vinduet, når brugeren har valgt en label

    # Start hovedløkken
    window.mainloop()

    # Returner det valgte label
    return selected_label.get()


rgb_labels = []
for i in range(1, 75):
    path = os.path.abspath(__file__+'/../../../') + f'\King Domino dataset\Cropped and perspective corrected boards\\{i}.jpg'
    image = cv.imread(path)

    # cv.imshow(f"image {i}", image)
    tiles = get_tiles(image)
    for x, row in enumerate(tiles):
        for y, tile in enumerate(row):
            winname = f"tile {x, y}"
            rgb_tile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
            r, g, b = np.median(rgb_tile, axis=(0, 1))
            print(f"image {i}", f"tile {x, y}", (r, g, b))

            label = display_image_with_buttons(rgb_tile)
            print(label)
            rgb_labels.append({"image": i, "tile": (x, y), "label": label, "rgb": (r, g, b)})
    df = pd.DataFrame(rgb_labels)
    df.to_csv("rgb_training.csv", index=False)

cv.waitKey()
cv.destroyAllWindows()
