import pickle
import numpy as np
import assemble_objects_rules
import plot_image
import matplotlib.pyplot as plt
from standard_values import *
from load_images import load_image
from numba import njit

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

    counts_detections_tiled = np.repeat(count_matrix_detections, num_annotations).reshape((num_detections, -1))
    counts_annotations_tiled = np.repeat(count_matrix_annotations, num_detections).reshape((num_detections, -1))

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
    counter = 0
    for annotation, detected_objects in evaluation_inputs:
        num_valid_annotations, num_valid_detections, IoU_values = extract_IoU_information(annotation, detected_objects)
        IoU_sum += sum(IoU_values)
        counter += len(IoU_values)
    return IoU_sum/counter

def evaluate_AP(evaluation_inputs):
    all_IoU_values = []
    num_all_valid_annotations = 0
    num_all_valid_detections = 0

    for annotation, detected_objects in evaluation_inputs:
        num_valid_annotations, num_valid_detections, IoU_values = extract_IoU_information(annotation, detected_objects)
        num_all_valid_detections += num_valid_detections
        num_all_valid_annotations += num_valid_annotations
        all_IoU_values.extend(IoU_values)

    for threshold in IoU_thresholds:
        points = []

        IoU_values_list = sorted([threshold] + [v for v in all_IoU_values if v > threshold])
        IoU_values_array = np.array(IoU_values_list)
        for value in IoU_values_list:
            IoU_over_value = IoU_values_array > value

            tp = np.sum(IoU_over_value)

            precision = tp / num_all_valid_detections
            recall = tp / num_all_valid_annotations
            points.append((precision, recall))
        a = np.asarray(points)
        plt.plot(a[:, 0], a[:, 1])
        plt.show()

def evaluate_model(model_type="Rules", eval_type="AP"):
    try:
        evaluation_inputs = load_evaluation_inputs(model_type)
    except:
        evaluation_inputs = save_evaluation_inputs(model_type)

    if eval_type == "AP":
        evaluate_AP(evaluation_inputs)

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
    evaluate_model()