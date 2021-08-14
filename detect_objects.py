import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tqdm
import os
import hashlib
import numpy as np
import tensorflow as tf
from yolo.yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from standard_values import *
from PIL import Image
from yolo.yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolo.yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def detect_objects(file):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=80, score_threshold=0.2)

    yolo.load_weights("yolo/checkpoints/yolov3.tf").expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open("yolo/data/coco.names").readlines()]
    logging.info('classes loaded')

    img_raw = tf.image.decode_image(
          open(f"Data/RGB/{file}", 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite("out.jpg", img)
    logging.info('output saved to: {}'.format("out.jpg"))

def build_example(name, class_map):
    img_raw = open(f"Data/RGB/{name}", 'rb').read()
    annotation = np.asarray(Image.open(f"Data/annotation/{name}"))

    key = hashlib.sha256(img_raw).hexdigest()
    n_objects = np.max(annotation)
    values = np.ones((n_objects, 4)) * -1
    for y in range(height):
        for x in range(width):
            index = annotation[y][x]
            if index != 0:
                if values[index - 1][0] < 0:
                    values[index - 1][0] = x
                    values[index - 1][1] = y
                    values[index - 1][2] = x
                    values[index - 1][3] = y
                else:
                    values[index-1][0] = min(values[index-1][0], x)
                    values[index-1][1] = min(values[index-1][1], y)
                    values[index-1][2] = max(values[index-1][2], x)
                    values[index-1][3] = max(values[index-1][3], y)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    for i in range(n_objects):
        difficult = False
        difficult_obj.append(int(difficult))

        xmin.append(float(values[i][0]) / width)
        ymin.append(float(values[i][1]) / height)
        xmax.append(float(values[i][2]) / width)
        ymax.append(float(values[i][3]) / height)
        classes_text.append("box".encode("utf8"))
        classes.append(class_map["box"])
        truncated.append(0)
        views.append("".encode("utf8"))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def create_dataset():
    class_map = {name: idx for idx, name in enumerate(
        open("yolo/data/boxes.names").read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter("yolo/data/boxes.tfrecord")
    image_list = os.listdir("Data/RGB")
    logging.info("Image list loaded: %d", len(image_list))
    for name in tqdm.tqdm(image_list):
        tf_example = build_example(name, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")

def visualize_dataset():
    class_names = [c.strip() for c in open("yolo/data/boxes.names").readlines()]
    logging.info('classes loaded')

    dataset = load_tfrecord_dataset("yolo/data/boxes.tfrecord", "yolo/data/boxes.names", 640)
    dataset = dataset.shuffle(3)

    for image, labels in dataset.take(1):
        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if x1 == 0 and x2 == 0:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(1)
            classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]

        logging.info('labels:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite("test_out.jpg", img)
        logging.info('output saved to: {}'.format("test_out.jpg"))


if __name__ == '__main__':
    visualize_dataset()
    quit()
    create_dataset()
    quit()
    detect_objects("test65.png")