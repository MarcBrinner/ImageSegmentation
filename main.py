import find_surfaces
import assemble_objects_CRF, assemble_objects_rules, assemble_objects_pairwise_clf
import evaluate_models as em
import config
from load_images import load_image
from plot_image import plot_surfaces

model_types = ["Rules", "Pairs", "CRF"]
clf_types = ["LR", "Neural", "Tree", "Forest"]
training_set_indices = config.train_indices
test_set_indices = config.test_indices

def train_pixel_similarity_function():
    find_surfaces.train_Gauss_clf()

def find_surfaces_for_dataset():
    find_surfaces.find_surfaces_for_indices(range(111), save_data=True, plot_result=False)

def train_surface_link_classifier(clf_type):
    assemble_objects_pairwise_clf.train_pairwise_classifier(clf_type)

def train_surface_link_classifier_through_interaction_model(clf_type):
    assemble_objects_CRF.train_CRF(clf_type)

def detect_objects_for_dataset(model_type="Pairs", indices=range(111), clf_type="Forest", use_CRF=True, do_post_processing=True):
    if model_type == model_types[0]:
        assemble_objects_rules.assemble_objects_for_indices(indices=indices, do_post_processing=do_post_processing)
    elif model_type == model_types[1]:
        assemble_objects_pairwise_clf.assemble_objects_for_indices(indices, use_CRF, clf_type, False, do_post_processing, True)
    elif model_type == model_types[2]:
        assemble_objects_CRF.assemble_objects_for_indices(indices, clf_type, do_post_processing, True)

def complete_segmentation(model_type="Pairs", index=0, clf_type="Forest", use_CRF=True, do_post_processing=True):
    if model_type == model_types[0]:
        model = assemble_objects_rules.get_full_prediction_model(do_post_processing=do_post_processing)
    elif model_type == model_types[1]:
        model = assemble_objects_pairwise_clf.get_full_prediction_model(clf_type, use_CRF, False, do_post_processing)
    elif model_type == model_types[2]:
        model = assemble_objects_CRF.get_full_prediction_model(clf_type, False, do_post_processing)
    else:
        print("Unknown model type.")
        quit()

    image_data = load_image(index)
    prediction = model(image_data)
    plot_surfaces(prediction["final_surfaces"])

def evaluate_models(model_type="Pairs", clf_type="Forest", use_CRF=True, do_post_processing=True):
    em.evaluate_model(model_type, {"clf_type": clf_type,
                                   "use_CRF": use_CRF,
                                   "do_post_processing": do_post_processing,
                                   "use_boxes": False})

if __name__ == '__main__':
    pass