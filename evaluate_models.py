import pickle
import numpy as np

import assemble_objects_CRF
import assemble_objects_pairwise_clf
import assemble_objects_rules
import plot_image
import matplotlib.pyplot as plt
from standard_values import *
from load_images import load_image
from numba import njit
from tqdm import tqdm

IoU_thresholds = [0.5, 0.75, 0.9]
min_annotation_overlap_threshold = 0.2
model_types = ["Rules"]

@njit()
def extract_IoU_information(annotation, detected_objects):
    num_detections = np.max(detected_objects)
    num_annotations = np.max(annotation)

    overlap_matrix = np.zeros((num_detections + 1, num_annotations + 1))
    count_matrix_detections = np.zeros(num_detections + 1)
    count_matrix_annotations = np.zeros(num_annotations + 1)

    for y in range(height):
        for x in range(width):
            a = annotation[y, x]
            o = detected_objects[y, x]

            count_matrix_detections[o] += 1
            count_matrix_annotations[a] += 1
            overlap_matrix[o, a] += 1

    overlap_matrix = overlap_matrix[1:, 1:]
    count_matrix_detections = count_matrix_detections[1:]
    count_matrix_annotations = count_matrix_annotations[1:]

    counts_detections_tiled = np.repeat(count_matrix_detections, num_annotations).reshape((num_detections, num_annotations))
    counts_annotations_tiled = np.transpose(np.repeat(count_matrix_annotations, num_detections).reshape((num_annotations, num_detections)))

    IoU = overlap_matrix / (counts_annotations_tiled + counts_detections_tiled - overlap_matrix)
    IoU_max_annotations = np.zeros(num_annotations)
    for i in range(num_annotations):
        for j in range(num_detections):
            if IoU[j, i] > IoU_max_annotations[i]:
                IoU_max_annotations[i] = IoU[j, i]

    detection_annotation_overlap = np.sum(overlap_matrix, axis=-1) / count_matrix_detections
    detection_annotation_overlap[np.isnan(detection_annotation_overlap)] = 0
    valid_detections = np.zeros(num_detections)
    valid_detections[detection_annotation_overlap > min_annotation_overlap_threshold] = 1

    valid_annotations = np.ones_like(count_matrix_annotations)
    valid_annotations[count_matrix_annotations == 0] = 0

    num_valid_detections = np.sum(valid_detections)
    num_valid_annotations = np.sum(valid_annotations)
    return num_valid_annotations, num_valid_detections, IoU_max_annotations

def evaluate_Avg_max_IoU(evaluation_inputs): # Alias Jaccard Index
    IoU_sum = 0
    for annotation, detected_objects in evaluation_inputs:
        num_valid_annotations, num_valid_detections, IoU_values = extract_IoU_information(annotation.astype("int64"), detected_objects)
        avg_max_IoU_sample = sum(IoU_values) / len(IoU_values)
        print(avg_max_IoU_sample)
        IoU_sum += avg_max_IoU_sample
    return IoU_sum/len(evaluation_inputs)

def evaluate_model(model_type="Rules", model_args={}):
    if model_type == "Rules":
        model = assemble_objects_rules.get_prediction_model()
    elif model_type == "Pairs" or model_type == "Pairs CRF":
        model_args["model_type"] = model_type
        model = assemble_objects_pairwise_clf.get_prediction_model(**model_args)
    elif model_type == "CRF":
        model = assemble_objects_CRF.get_prediction_model(**model_args)

    evaluation_inputs = []
    for index in tqdm(test_indices):
        image_data = load_image(index)
        prediction = model(image_data, index)
        evaluation_inputs.append((image_data[2], prediction["final_surfaces"]))

    score = evaluate_Avg_max_IoU(evaluation_inputs)
    print(score)

def save_evaluation_inputs(model_type="Rules"):
    if model_type not in model_types:
        print("Model type not recognized.")
        quit()

    if model_type == "Rules":
        model = assemble_objects_rules.get_prediction_model()

    evaluation_inputs = []
    for index in test_indices:
        image_data = load_image(index)
        prediction = model(image_data)
        evaluation_inputs.append((image_data[2], prediction["final_surfaces"]))

    pickle.dump(evaluation_inputs, open(f"out/evaluation_inputs/{model_type}.pkl", "wb"))
    return evaluation_inputs

def load_evaluation_inputs(model_type="Rules"):
    if model_type not in model_types:
        print("Model type not recognized.")
        quit()
    inputs = pickle.load(open(f"out/evaluation_inputs/{model_type}.pkl", "rb"))
    return inputs

if __name__ == '__main__':
    evaluate_model(model_type="CRF", model_args={"clf_type": "Neural"})