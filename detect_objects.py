import os
import tensorflow as tf
import numpy as np
import load_images
import plot_image
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

import standard_values
from standard_values import *
from core.yolov3 import YOLOv3, decode, compute_loss
from core import utils, dataset, config
from PIL import Image
input_size = 416

def get_object_detector_model(weights="./parameters/YOLO_trained_weights/weights.ckpt"):
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    if weights:
        model.load_weights(weights)
        #utils.load_weights(model, weights)
    model.compile(loss=tf.keras.losses.MAE, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=[], run_eagerly=True)

    #model.summary()
    return model


def detect_objects(model, input_image):
    input_image_size = input_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(input_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict_on_batch(image_data)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, input_image_size, input_size, 0.5)
    bboxes = utils.nms(bboxes, 0.5, method='nms')
    return bboxes

def show_bounding_boxes(input_image, bboxes):
    image = utils.draw_bbox(input_image.copy(), bboxes)
    plot_image.plot_array_PLT(image)

def get_object_detector(weights=None):
    if weights:
        model = get_object_detector_model(weights)
    else:
        model = get_object_detector_model()
    return lambda x: detect_objects(model, x)

def main():
    detector = get_object_detector_model()
    for i in standard_values.test_indices:
        _, rgb, _  = load_images.load_image(i)
        bboxes = detect_objects(detector, rgb)
        show_bounding_boxes(rgb, bboxes)

if __name__ == '__main__':
    main()
