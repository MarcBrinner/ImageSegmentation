import tensorflow as tf
import numpy as np
import load_images
import plot_image
from core.yolov3 import YOLOv3, decode
from core import utils
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
    #model2 = tf.keras.Model(model.layers[0].input, model.layers[6].output)
    model.compile(loss=tf.keras.losses.MAE, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=[], run_eagerly=False)

    model.summary()
    return model

def detect_objects(model, input_image):
    input_image_size = input_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(input_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict_on_batch(image_data)
    print([np.any(np.isnan(pred_bbox[i])) for i in range(3)])

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, input_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.1, method='nms')
    return bboxes

def show_bounding_boxes(input_image, bboxes):
    image = utils.draw_bbox(input_image.copy(), bboxes)
    plot_image.plot_array_PLT(image)

def main():
    detector = get_object_detector()
    for i in range(110):
        _, rgb, _  = load_images.load_image(i)

        bboxes = detect_objects(detector, rgb)
        #show_bounding_boxes(rgb, bboxes)
        bboxes = detect_objects(detector, rgb)
        #show_bounding_boxes(rgb, bboxes)
        bboxes = detect_objects(detector, rgb)
        #show_bounding_boxes(rgb, bboxes)
        bboxes = detect_objects(detector, rgb)
        quit()
        show_bounding_boxes(rgb, bboxes)
        bboxes = detect_objects(detector, rgb)
        show_bounding_boxes(rgb, bboxes)
        bboxes = detect_objects(detector, rgb)
        show_bounding_boxes(rgb, bboxes)
        bboxes = detect_objects(detector, rgb)
        show_bounding_boxes(rgb, bboxes)

if __name__ == '__main__':
    main()
