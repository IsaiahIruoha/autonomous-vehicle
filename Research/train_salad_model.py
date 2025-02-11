#!/usr/bin/python3.9
import numpy as np
import os
import sys

print("version", sys.version)
print("cwd",)

from os import listdir
print(listdir(os.getcwd()))

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

# Model Maker requires that we load our dataset using the DataLoader API. So 
# in this case, we'll load it from a CSV file that defines 175 images for 
# training, 25 images for validation, and 25 images for testing.
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv')

print(f'train count: {len(train_data)}')
print(f'validation count: {len(validation_data)}')
print(f'test count: {len(test_data)}')

spec = object_detector.EfficientDetLite0Spec()
model = object_detector.create(train_data=train_data,
                               model_spec=spec,
                               validation_data=validation_data,
                               epochs=50,
                               batch_size=10,
                               train_whole_model=True)

# Now we'll use the test dataset to evaluate how well the model performs with 
# data it has never seen before (the test subset). The evaluate() method 
# provides output in the style of COCO evaluation metrics:
print('Evaluating model on test data...')
model.evaluate(test_data)

# Next, we'll export the model to the TensorFlow Lite format. By default, the 
# export() method performs full integer post-training quantization, which is 
# exactly what we need for compatibility with the Edge TPU. (Model Maker uses 
# the same dataset we gave to our model spec as a representative dataset, 
# which is required for full-int quantization.)

# We just need to specify the export directory and format. By default, it 
# exports to TF Lite, but we also want a labels file, so we declare both:
TFLITE_FILENAME = 'efficientdet-lite-salad.tflite'
LABELS_FILENAME = 'salad-labels.txt'
print('Exporting model to tflite model...')
model.export(export_dir='.', tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])


# Exporting the model to TensorFlow Lite can affect the model accuracy, due to 
# the reduced numerical precision from quantization and because the original 
# TensorFlow model uses per-class non-max supression (NMS) for post-processing, 
# while the TF Lite model uses global NMS, which is faster but less accurate.

# Therefore you should always evaluate the exported TF Lite model and be sure 
# it still meets your requirements:
print('Evaluating tflite model on test_data...')
model.evaluate_tflite(TFLITE_FILENAME, test_data)


