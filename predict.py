#!/usr/local/bin/python3
#import turicreate as tc
import sys
import os
import re
import image_labeller as model2

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def predict_model_A(img_paths):
    print('turicreate')
    model = tc.load_model('models/512x384v4.model')

    input_paths = tc.SArray(img_paths[0])
    images = tc.SArray([tc.Image(path) for path in img_paths[0]])
    output_paths = tc.SArray(img_paths[1])

    data = tc.SFrame({
        'input': input_paths,
        'image': images,
        'output': output_paths
        })

    predictions = model.predict(data)

    ddata = tc.SFrame({
        'input': input_paths,
        'image': images,
        'predictions': predictions,
        'output': output_paths
        })

    ddata['label'] = tc.object_detector.util.draw_bounding_boxes(ddata['image'], ddata['predictions'])

    for record in ddata:
        record['label'].save(record['output'])

def predict_model_B(dirs):
    #input_files, output_dirs = dirs
    for file, d in dirs.transpose():
        model2.label(file, d)
    
def intersection_over_union(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou 

def main(model_path, data_path):
    model = tc.load_model(model_path)
    data = tc.SFrame(data_path)
    train_data, test_data = data.random_split(0.8, seed=7)
    predictions = model.predict(test_data)
    test = tc.SFrame({'image': test_data["image"], 'predictions': predictions})
    test['image_with_predictions'] = tc.object_detector.util.draw_bounding_boxes(test['image'], test['predictions'])
    test.explore()

if __name__=='__main__':
    if len(sys.argv) != 3:
        print('error')
        sys.exit(-1)
    main(sys.argv[2], sys.argv[1])
