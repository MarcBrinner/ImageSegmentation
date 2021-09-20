import os
import tensorflow as tf
import numpy as np
import load_images
import plot_image
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from standard_values import *
from core.yolov3 import YOLOv3, decode, compute_loss
from core import utils, dataset, config
from PIL import Image
input_size = 416

def get_object_detector(weights="./parameters/yolov3.weights"):
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    if weights:
        utils.load_weights(model, weights)
    model.compile(loss=tf.keras.losses.MAE, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=[], run_eagerly=True)

    model.summary()
    return model

def create_training_annotation_file():
    image_names = os.listdir("data/RGB")
    annotation_file_string = ""
    for name in tqdm(image_names):
        annotation_string = f"data/RGB/{name} "
        annotation = np.asarray(Image.open(f"data/annotation/{name}"))
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
    with open("data/YOLO_training_annotation.txt", "w+") as file:
        file.write(annotation_file_string)

def visualize_dataset():
    with open("data/YOLO_training_annotation.txt", "r") as annotation_file:
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

def train_detector():
    model = get_object_detector()
    trainset = dataset.Dataset('train')
    logdir = "./data/log"
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = config.cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = config.cfg.TRAIN.EPOCHS * steps_per_epoch

    input_tensor = tf.keras.layers.Input([416, 416, 3])
    conv_tensors = YOLOv3(input_tensor)

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * config.cfg.TRAIN.LR_INIT
            else:
                lr = config.cfg.TRAIN.LR_END + 0.5 * (config.cfg.TRAIN.LR_INIT - config.cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    for epoch in range(config.cfg.TRAIN.EPOCHS):
        for image_data, target in trainset:
            train_step(image_data, target)
        model.save_weights("parameters/YOLO_trained_weights")

def detect_objects(model, input_image):
    input_image_size = input_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(input_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict_on_batch(image_data)
    print([np.any(np.isnan(pred_bbox[i])) for i in range(3)])

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, input_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.9, method='nms')
    return bboxes

def show_bounding_boxes(input_image, bboxes):
    image = utils.draw_bbox(input_image.copy(), bboxes)
    plot_image.plot_array_PLT(image)

def main():
    detector = get_object_detector()
    for i in range(110):
        _, rgb, _  = load_images.load_image(i)
        bboxes = detect_objects(detector, rgb)
        show_bounding_boxes(rgb, bboxes)

if __name__ == '__main__':
    #train_detector()
    #quit()
    create_training_annotation_file()
    quit()
    #visualize_dataset()
    #quit()
    main()
