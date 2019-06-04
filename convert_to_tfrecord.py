#!/usr/local/bin/python3

import json
import sys
import random
import os.path
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image

def read(path):
    with open(path) as f:
        return json.loads(f.read())

def transform_bbox(bbox):
##    width = bbox[2]
##    height = bbox[3]
##    x_min, y_min = bbox[:2]
##    x_max = x_min + width
##    y_max = y_min + height
##    return x_min, x_max, y_min, y_max
    return bbox[0], bbox[4], bbox[1], bbox[5]


def path_for_image_name(name):
    return os.path.dirname(dataset_dir) + '/' + name

def create_tf_example(image, annotations):
    file = image['file_name']
    filename = bytes(file, 'utf-8')
    with open('./datasets/512x384/'+file, 'rb') as f:
        encoded_image_data = f.read()
    image_format = b'jpeg'
    with Image.open('./datasets/512x384/'+file) as im:
        width, height = im.size
    print(file, width, height)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for annot in annotations:
        x_min, x_max, y_min, y_max = transform_bbox(*annot['segmentation'])
        x_min /= width
        x_max /= width
        y_min /= height
        y_max /= height
        xmins.append(max(0, x_min))
        xmaxs.append(min(1, x_max))
        ymins.append(max(0, y_min))
        ymaxs.append(min(1, y_max))
        classes_text.append(b'bottle')
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_image_data),
          'image/format': dataset_util.bytes_feature(image_format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

path = './datasets/512x384/dataset.json'
data = read(path)
dataset_dir = './datasets/512x384/'

examples = []

images = data['images']
a = [[] for _ in range(len(images))]

this = []
id_ = None
for annot in data['annotations']:
    if annot['ignore'] == 1:
        print('ignore')
        continue
    if annot['image_id'] != id_:
        if this:
            examples.append(create_tf_example(data['images'][id_-1], this))
        this = [annot]
        id_ = annot['image_id']
    else:
        this.append(annot)

rand = list(examples)
random.shuffle(rand)
split = len(rand)//10
test = rand[:split]
train = rand[split:]

writer = tf.python_io.TFRecordWriter('./data/train.record')
for example in train:
    writer.write(example.SerializeToString())
writer.close()

writer = tf.python_io.TFRecordWriter('./data/test.record')
for example in test:
    writer.write(example.SerializeToString())
writer.close()
