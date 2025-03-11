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

from split_dataset import split_dataset

dataset_is_split = False

# Your labels map as a dictionary (zero is reserved):
label_map = {1: 'trafficlight', 2: 'stop', 3: 'speedlimit', 4: 'crosswalk'}

# Specify the path to all images and annotations.
images_in = '/content/dataset/images'
annotations_in = '/content/dataset/annotations'

# Unlike the salads dataset which defined which data was part of the test 
# and validation sets, the road signs dataset doesn't do this.  Thus, we use 
# the split_dataset function (found int split_dataset.py) to help us randomly 
# split the dataset test (20% of the data), validation (20%), and training 
# (60%) sets.  
train_dir, val_dir, test_dir = split_dataset(images_in, annotations_in,
                                              val_split=0.2, test_split=0.2,
                                              out_path='split-dataset')

# We need to instantiate a separate DataLoader for each split dataset
train_data = object_detector.DataLoader.from_pascal_voc(
    os.path.join(train_dir, 'images'),
    os.path.join(train_dir, 'annotations'), label_map=label_map)
validation_data = object_detector.DataLoader.from_pascal_voc(
    os.path.join(val_dir, 'images'),
    os.path.join(val_dir, 'annotations'), label_map=label_map)
test_data = object_detector.DataLoader.from_pascal_voc(
    os.path.join(test_dir, 'images'),
    os.path.join(test_dir, 'annotations'), label_map=label_map)

print(f'train count:      {len(train_data)}')
print(f'validation count: {len(validation_data)}')
print(f'test count:       {len(test_data)}')

# Model Maker supports the EfficientDet-Lite family of object detection models 
# that are compatible with the Edge TPU. (EfficientDet-Lite is derived from 
# EfficientDet, which offers state-of-the-art accuracy in a small model size). 
# There are several model sizes you can choose from:

# 	Model architecture 	Size(MB)* 	Latency(ms)** 	Average Precision***
# 	EfficientDet-Lite0 	5.7 	    37.4 	        30.4%
# 	EfficientDet-Lite1 	7.6 	    56.3 	        34.3%
# 	EfficientDet-Lite2 	10.2 	    104.6 	        36.0%
# 	EfficientDet-Lite3 	14.4 	    107.6 	        39.4%
	
# * File size of the compiled Edge TPU models.
# ** Latency measured on a desktop CPU with a Coral USB Accelerator.
# *** Average Precision is the mAP (mean Average Precision) on the COCO 2017 
# validation dataset. 				

# Beware that the Lite2 and Lite3 models do not fit onto the Edge TPU's 
# onboard memory, so you'll see even greater latency when using those, due to 
# the cost of fetching data from the host system memory. Maybe this extra 
# latency is okay for your application, but if it's not and you require the 
# precision of the larger models, then you can pipeline the model across 
# multiple Edge TPUs (more about this when we compile the model below).

# For this tutorial, we'll use Lite0:

spec = object_detector.EfficientDetLite0Spec()

# The EfficientDetLite0Spec constructor also supports several arguments that 
# specify training options, such as the max number of detections (default is 
# 25 for the TF Lite model) and whether to use Cloud TPUs for training. You 
# can also use the constructor to specify the number of training epochs and 
# the batch size, but you can also specify those in the next step.

# Now we need to create our model according to the model spec, load our dataset 
# into the model, specify training parameters, and begin training.
# Using Model Maker, we accomplished all of that with create():
model = object_detector.create(train_data=train_data,
                               model_spec=spec,
                               validation_data=validation_data,
                               epochs=50,
                               batch_size=10,
                               train_whole_model=True)

# Now we'll use the test dataset to evaluate how well the model performs with 
# data it has never seen before. The evaluate() method provides output in the 
# style of COCO evaluation metrics. 

print('Evaluating model on test data.')
results = model.evaluate(test_data)
print("model evaluation results: ")
print(results)


# Because the default batch size for EfficientDetLite models is 64, this needs 
# only 1 step to go through all 25 images in the salad test set. You can also 
# specify the batch_size argument when you call evaluate().

# Next, we'll export the model to the TensorFlow Lite format. By default, the 
# export() method performs full integer post-training quantization, which is 
# exactly what we need for compatibility with the Edge TPU. (Model Maker uses 
# the same dataset we gave to our model spec as a representative dataset, which 
# is required for full-int quantization.) We just need to specify the export 
# directory and format. By default, it exports to TF Lite, but we also want a 
# labels file, so we declare both:

TFLITE_FILENAME = 'efficientdet-lite-road-signs.tflite'
LABELS_FILENAME = 'road-signs-labels.txt'

model.export(export_dir='.', tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

# Exporting the model to TensorFlow Lite can affect the model accuracy, due to 
# the reduced numerical precision from quantization and because the original 
# TensorFlow model uses per-class non-max supression (NMS) for post-processing, 
# while the TF Lite model uses global NMS, which is faster but less accurate.
# Therefore you should always evaluate the exported TF Lite model and be sure 
# it still meets your requirements:

print('Evaluating tflite model on test data.')
results = model.evaluate_tflite(TFLITE_FILENAME, test_data)
print("tflite model evaluation results: ")
print(results)