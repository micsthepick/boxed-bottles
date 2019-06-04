#!/usr/local/bin/python3

import argparse
import os
import numpy as np
import predict as models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def chomp(sequence):
    s = str(sequence)
    return (s[:75] + '..') if len(s) > 75 else data

def isempty(path):
    return os.path.exists(path) and os.path.isdir(path) and not os.listdir(path)

def pngOrJpg(name):
    k = name.lower()
    return k.endswith('.png') or k.endswith('.jpg') or k.endswith('.jpeg')

def predict(args):
    src, out, algor = args.input, args.output, args.algorithm

    imgs = os.listdir(src)
    print(f"Input Dir: {src}")
    print(f"Source Images: {chomp(imgs)}")
    print(f"Output Dir: {out}")
    print(f"Using Algorithm: {algor}")

    use_model_a = algor == 'tc' or algor == 'turicreate'
    render_prediction = models.predict_model_A if use_model_a else models.predict_model_B

    if not os.path.exists(out):
        os.makedirs(out)
    
    if not isempty(out):
        print(f"refusing to write to output directory - directory is not empty")

    imgs = list(filter(pngOrJpg, imgs))
    img_paths = np.array((np.array([os.path.join(src, img) for img in imgs]), np.array([os.path.join(out, img) for img in imgs])))
    render_prediction(img_paths)

    print("Predictions Complete")

def metrics(args):
    src, algor, model = args.input, args.algorithm, args.model
    imgs = os.listdir(src)
    
    print(f"Input Dir: {src}")
    print(f"Source Images: {chomp(imgs)}")
    print(f"Using Algorithm: {algor}")
    print(f"Model Path: {model}")

    use_model_a = algor == 'tc' or algor == 'turicreate'
    compute_predictions = models.compute_predictions_model_A if use_model_a else models.compute_predictions_model_B

    imgs = list(filetr(pngOrJpg, imgs))
    img_paths = np.array([os.path.join(src, img) for img in imgs])

    predictions = compute_predictions(img_paths)
    

    models.intersection_over_union(

    print("Metrics Complete")

if __name__=='__main__':

    #np.warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='bottle detector driver.')

    subparsers = parser.add_subparsers(help='subcommands')

    predictionParser = subparsers.add_parser('predict', help='make predictions')
    predictionParser.set_defaults(func=predict)
    predictionParser.add_argument(
            '--input', 
            dest='input', 
            type=str, 
            required=True,
            help='directory of arbitary images')
    predictionParser.add_argument(
            '--output', 
            dest='output', 
            type=str, 
            required=True,
            help='directory to put images with predictions drawn on')
    predictionParser.add_argument(
            '--algorithm', 
            dest='algorithm', 
            type=str, 
            choices=['turicreate', 'tc', 'tensorflow', 'tf'],
            required=True,
            help='selects the model algorithm to perform predictions with')

    metricsParser = subparsers.add_parser('metrics', help='compute metrics for a given model')
    metricsParser.set_defaults(func=metrics)
    metricsParser.add_argument(
            '--input', 
            dest='input', 
            type=str, 
            required=True,
            help='directory of arbitary images')
    metricsParser.add_argument(
            '--algorithm', 
            dest='algorithm', 
            type=str, 
            required=True,
            choices=['turicreate', 'tc', 'tensorflow', 'tf'],
            help='selects the model algorithm')
    metricsParser.add_argument(
            '--model', 
            dest='model', 
            type=str, 
            required=True,
            help='point to a path containing a model')

    args = parser.parse_args()
    args.func(args)
