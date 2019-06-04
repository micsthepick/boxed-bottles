#!/usr/local/bin/python3

from os import getcwd, mkdir
from os.path import basename, splitext, dirname, relpath, exists
from glob import glob
from PIL import Image, ExifTags
from sys import exit

output_folder = getcwd() + "/datasets/resized/"
size = (1024, 768)

if not exists(output_folder):
    mkdir(output_folder)

def rename(path):
    return output_folder + relpath(path).replace('/', '-')

def generate_thumbnail(path, new_path, size):
    try:
        image=Image.open(path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

        image.thumbnail(size, Image.LANCZOS) #ANTIALIAS)
        image.save(new_path)
        image.close()
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
if __name__=='__main__':
    for path in glob(getcwd() + "/Photos/Bottles/MichaelasBottles/*.JPG"):
        if output_folder in path:
            continue
        print(f"processing: {relpath(path)}")
        new_path = rename(path)
        if exists(new_path):
            print("error: would over write path")
            print(f"path: {relpath(new_path)}")
            exit(-1)
        generate_thumbnail(path, new_path, size)
        print(f"thumbdnail generate at: {relpath(new_path)}")
