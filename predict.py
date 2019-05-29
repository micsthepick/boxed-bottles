#!/usr/local/bin/python3
import turicreate as tc
import sys

if len(sys.argv) != 3:
    print('error')
    sys.exit(-1)

model = tc.load_model(sys.argv[2])

data = tc.SFrame(sys.argv[1])

train_data, test_data = data.random_split(0.8, seed=7)

predictions = model.predict(test_data)

test = tc.SFrame({'image': test_data["image"], 'predictions': predictions})
test['image_with_predictions'] = tc.object_detector.util.draw_bounding_boxes(test['image'], test['predictions'])
test[['image', 'image_with_predictions']].explore()
