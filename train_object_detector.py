#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import plot_image
import core.utils as utils
import config
from config import *
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg
from PIL import Image

def train(load_trained_weights=False):
    trainset = Dataset('train')
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

    input_tensor = tf.keras.layers.Input([416, 416, 3])
    conv_tensors = YOLOv3(input_tensor)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, i)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_tensor, output_tensors)
    if load_trained_weights:
        model.load_weights("./parameters/YOLO_trained_weights/weights.ckpt")
    else:
        utils.load_weights(model, "parameters/yolov3.weights")
    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                              giou_loss, conf_loss,
                                                              prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

    for epoch in range(cfg.TRAIN.EPOCHS):
        trainset.shuffle()
        for image_data, target in trainset:
            train_step(image_data, target)
        model.save_weights("./parameters/YOLO_trained_weights/weights.ckpt")

def create_annotation_files():
    annotation_file_string = ""
    for set_type in ["train", "test"]:
        for index in config.train_indices if set_type == "train" else config.test_indices:
            print(index)
            annotation_string = f"data/RGB/{index}.png "
            annotation = np.asarray(Image.open(f"data/annotation/{index}.png"))
            num_objects = np.max(annotation)
            mins = np.ones((num_objects, 2), dtype="int32") * np.inf
            maxs = np.ones((num_objects, 2), dtype="int32") * -np.inf
            for y in range(height):
                for x in range(width):
                    if (i := annotation[y][x]) != 0:
                        values = np.asarray([x, y], dtype="int32")
                        mins[i-1] = np.minimum(mins[i-1], values)
                        maxs[i-1] = np.maximum(maxs[i-1], values)
            for i in range(num_objects):
                try:
                    annotation_string = annotation_string + f"{int(mins[i][0])},{int(mins[i][1])},{int(maxs[i][0])},{int(maxs[i][1])},73 "
                except:
                    pass
            annotation_file_string = annotation_file_string + annotation_string + "\n"
        with open(f"data/YOLO_annotation_files/YOLO_annotation_{set_type}.txt", "w+") as file:
            file.write(annotation_file_string)

def visualize_dataset(set_type="train"):
    with open(f"data/YOLO_annotation_files/YOLO_annotation_{set_type}.txt", "r") as annotation_file:
        for line in annotation_file:
            parts = line.split(" ")
            del parts[-1]
            image = np.asarray(Image.open(parts[0]))
            boxes = np.zeros((len(parts)-1, 6))
            for i in range(len(parts)-1):
                #print(parts[i+1].split(","))
                numbers = parts[i+1].split(",")
                boxes[i] = np.asarray([*[float(x) for x in numbers[:-1]], 0.5, numbers[-1]])
            image = utils.draw_bbox(image, boxes)
            plot_image.plot_array_PLT(image)

if __name__ == '__main__':
    pass