#!/usr/local/bin/python3

import json
import sys
import turicreate as tc
import os.path

def read(path):
    with open(path) as f:
        return json.loads(f.read())

def transform_bbox(bbox):
    width = bbox[2]
    height = bbox[3]
    x = bbox[0] + width/2
    y = bbox[1] + height/2
    return { 'x': x, 'y': y, 'width': width, 'height': height }

def import_annotations(images, annotations):
    a = []
    for _ in range(len(images)):
        a.append([])

    for annotation in annotations:
        if annotation['ignore'] == '1':
            continue
        print(f'processing image id: {annotation["image_id"]}')
        bbox = transform_bbox(annotation['bbox'])
        img_id = int(annotation['image_id']) 
        a[img_id - 1].append({ 'coordinates': bbox, 'label': 'bottle' })

    return tc.SArray(a)

def path_for_image_name(name):
    return os.path.dirname(dataset_dir) + '/' + name

def import_images(images):
    images = sorted(images, key=lambda img: img['id'])
    image_ids = [i['id'] for i in images]
    file_name = [i['file_name'] for i in images]
    imgs = [tc.Image(path_for_image_name(i['file_name'])) for i in images]
    return tc.SFrame( {
        'id': tc.SArray(image_ids),
        'file_name': tc.SArray(file_name),
        'image': tc.SArray(imgs)
        })

if __name__=='__main__':
    if len(sys.argv) == 3:
        path = sys.argv[1]
        data = read(path)
        dataset_dir = path

        images = import_images(data['images'])
        annotations = import_annotations(images, data['annotations'])
        frames = images.add_column(annotations, 'annotations')
        frames.save(sys.argv[2])

    else:
        print(f"error: invalid command")
        print(f"{sys.argv[0]}: <path to dataset.json> <sframes output dir>")


