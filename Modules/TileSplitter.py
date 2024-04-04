#Andreas' tilesplitter funktion
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tile = image[y * 100:(y + 1) * 100, x * 100:(x + 1) * 100]
            tiles[-1].append(tile)
    return tiles