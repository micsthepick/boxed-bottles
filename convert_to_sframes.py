#!/usr/local/bin/python3

import json
import sys
import turicreate as tc
import os.path

def read(path):
    with open(path) as f:
        return json.loads(f.read())

def convert(data, columns):
    def extra(key):
        return tc.SArray([d[key] for d in data])
    return tc.SFrame(dict(zip(columns, map(extra, columns))))

def make_images(path, metadata, bboxes):
    imgs = dict([(i['id'], (tc.Image(path + '/' + i['file_name']), i['file_name'])) for i in metadata])

    images = map(lambda img_id: imgs[img_id][0], bboxes['image_id'])
    paths = map(lambda img_id: imgs[img_id][1], bboxes['image_id'])

    return (tc.SArray(images), tc.SArray(paths))

def extra_bboxes(bboxes):
    print(bboxes[0])
    def t(bbox):
        x = bbox[0] - bbox[2]/2
        y = bbox[1] - bbox[3]/2
        width = bbox[2]
        height = bbox[3]
        return {'coordinates': {'x': x, 'y': y, 'width': width, 'height': height }, 'label': 'trash'}
    return tc.SArray(list(map(t, bboxes)))

if __name__=='__main__':
    if len(sys.argv) == 3:
        path = sys.argv[1]
        data = read(path)

        image_metadata = convert(data['images'], ['file_name', 'id'])
        bboxes = convert(data['annotations'], ['area', 'image_id', 'bbox', 'id', 'ignore'])
        images, paths = make_images(os.path.dirname(path), image_metadata, bboxes)

        bbboxes = extra_bboxes(bboxes['bbox'])
        frames = bboxes.add_column(images, 'images').add_column(paths, 'file_name').add_column(bbboxes, 'annotation')
    else:
        print(f"error: invalid command")
        print(f"{sys.argv[0]}: <path to dataset.json> <sframes output dir>")


