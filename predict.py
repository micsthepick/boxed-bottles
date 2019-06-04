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
    print(dirs.shape)
    for file, d in dirs.transpose():
        model2.label(file, d)
    

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
