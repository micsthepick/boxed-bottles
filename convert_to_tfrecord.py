#!/usr/local/bin/python3

import json
import sys
import random
import os.path
import tensorflow as tf
from object_detection.utils import dataset_util

def read(path):
    with open(path) as f:
        return json.loads(f.read())

def transform_bbox(bbox):
    width = bbox[2]
    height = bbox[3]
    x_min, y_min = bbox[:2]
    x_max = x_min + width
    y_max = y_min + height
    return x_min, x_max, y_min, y_max


def path_for_image_name(name):
    return os.path.dirname(dataset_dir) + '/' + name

def create_tf_example(image, annotations):
    height = image['height']
    width = image['width']
    file = image['file_name']
    filename = bytes(file, 'utf-8')
    with open('./datasets/512x384/'+file, 'rb') as f:
        encoded_image_data = f.read()
    image_format = b'jpeg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for annot in annotations:
        x_min, x_max, y_min, y_max = transform_bbox(annot['bbox'])
        xmins.append(x_min/width)
        xmaxs.append(x_max/width)
        ymins.append(y_min/height)
        ymaxs.append(y_max/height)
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
id_ = None
this = []
for annot in data['annotations']:
    if annot['id'] != id_:
        id_ = annot['id']
        if this:
            examples.append(create_tf_example(data['images'][id_-1], this))
            this = []
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
