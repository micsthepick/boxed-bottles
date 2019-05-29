#!/usr/local/bin/python3
import turicreate as tc
import sys
import os

if len(sys.argv) != 3: 
    print("invalid argument")
    sys.exit(-1)

topk = int(sys.argv[1])
path = sys.argv[2]

if topk == 0:
    data = tc.SFrame(path)
else:
    data = tc.SFrame(path).topk('id', topk)
data['images_with_ground_truth'] = tc.object_detector.util.draw_bounding_boxes(data['image'], data['annotations'])
data.explore()
