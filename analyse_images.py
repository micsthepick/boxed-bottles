#!/usr/local/bin/python3

from os import getcwd
from glob import glob
from PIL import Image

dimensions = dict()
total_images = 0

for path in glob(getcwd() + "Photos/Bottles/**/*.JPG"):
    img = Image.open(path)
    width, height = img.size
    dims = f"{width}x{height}"
    count = dimensions.get(dims, 0)
    dimensions[dims] = count + 1
    total_images += 1

for (key, value) in dimensions.items():
    print(f"dimension: {key}, count: {value}")


print(f"total images: {total_images}")
