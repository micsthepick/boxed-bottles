#!/usr/local/bin/python3
import turicreate as tc
import sys

if len(sys.argv) != 4:
    print("invalid arguments")
    sys.exit(-1)

iterations = int(sys.argv[1])
path = sys.argv[2]
model_path = sys.argv[3]

# Load the data
data =  tc.SFrame(path)

# Make a train-test split
train_data, test_data = data.random_split(0.8, seed=7)

# Create a model
model = tc.object_detector.create(train_data, annotations="annotations", feature="image", max_iterations=iterations)

# Save the model for later use in Turi Create
model.save(model_path)

