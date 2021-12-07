# ImageSegmentation
## Requirements
This project uses python 3.8 with tensorflow 2.6.0 and tensorflow-addons 0.14.0.
## File Overview:
- `assemble_objects_CRF.py`: Contains methods to assemble the detected surfaces using
the interaction model and pairwise classifiers for pairs of surfaces that are trained
through the interaction model.
- `assemble_objects_pairwise_clf.py`: Contains methods to assemble the detected surfaces using
the pairwise classifiers, either by directly using the pairwise prediction as the final link
prediction or by additionally using the interaction model. This file also contains the 
methods to train the pairwise classifiers outside the interaction model.
- `assemble_objects_rules.py`: Contains methods to assemble the detected surfaces using
the rule-based system.
- `config.py`: Contains parameters that can be set for different parts of the segmentation algorithm.
- `CRF_tools.py`: Contains useful methods for working with the interaction model, which
are used by `assemble_objects_CRF.py` and `assemble_objects_pairwise_clf.py`, because both use
the interaction model, once with the classifiers inside it and once with separate classifiers.
- `detect_objects.py`: Contains the methods used for object detection with the YoloV3 object detector.
- `evaluate_models.py`: Contain the methods used to calculate the AvgIoU score for all models.
- `find_surfaces.py`: Contains all methods needed to detect the smooth surfaces patches and
assign additional pixels to them (with the CRF) to detect complete surfaces. This file also
contains a method to train the pairwise pixel similarity function used in the CRF.
- `image_processing_models_GPU.py`: This file contains some tensorflow models for fast image processing
on the GPU.
- `load_images.py`: This file contains all methods required to load different kinds of images or
saved image data.
- `main.py`: This file contains a range of methods to perform all major tasks in this project,
from training the individual models to segmenting images and evaluating all models on the test set.
- `plot_image.py`: This file contains methods for plotting images and detected surfaces in different ways.
- `post_processing.py`: This file contains everything related to the post-processing procedure for
complete surfaces.
- `process_surfaces.py`: Many different methods in other files all need to use the same methods to
extract data from the surfaces detected in the first steps, or to process them in some other way.
This file contains all methods related to the processing of surfaces for easy access by other files.
- `train_object_detector.py`: The methods needed to train the YoloV3 object detector on the given
training set.
- `utils.py`: Contains methods that needed to be used at different points in the project.

## Testing the Project
The `main.py` file contains all methods needed to test the project. These methods are the following:
- `train_pixel_similarity_function()`: This function trains the pixel similarity function that
is used for assigning additional pixels to the detected surface patches. A corresponding training data
set is needed, which will be created if it is not found. Exemplary trained parameters are already
included in this project.
- `find_surfaces_for_dataset()`: This function detects the smooth surfaces for the whole data set
  (the Object Segmentation Database, 110 images) and saves the results in the `/out` folder of the project.
Doing this is required for some later steps, especially if a training set for joining surfaces needs to be
created. It can happen that a resources exhausted error occurs, which probably happens due to a
memory leakage bug from tensorflow. If this is the case, just restart the procedure beginning from the last
index that was printed out (by replacing the `range(111)` by `range(start_index, 111)`).
- `train_surface_link_classifier()`: This method allows to train the pairwise classifiers that predict
the link probability for a pair of surfaces. This requires a corresponding training set, if it is not
present it will be created first. This requires the surfaces for all images from the data set to
be found and saved, which can be done using the `find_surfaces_for_dataset()` method. The parameter
`clf_type` specifies the classifier that shall be trained, and it has to take one of the values
from the `clf_types` list.
- `train_surface_link_classifier_through_interaction_model()`: This method allows to train the pairwise classifiers that predict
the link probability for a pair of surfaces through the interaction model. This requires a corresponding training set, if it is not
present it will be created first. This requires the surfaces for all images from the data set to
be found and saved, which can be done using the `find_surfaces_for_dataset()` method. The parameter
`clf_type` specifies the classifier that shall be trained, and it has to take one of the values
from the `clf_types` list that corresponds to a differentiable classifier (`"LR"` or `"Neural"`).
- `detect_objects_for_dataset()`: This method allows producing segmentations for the whole data set
and plotting them to visualize the performance of the algorithms. This requires all surfaces
for the data set to be already detected and saved, which can be done by running the `find_surfaces_for_dataset()`
method. There are several parameters,
some of which are only needed for certain model types. These are the parameters that can be set:
  - `model_type`: Has to be one of the model types present in the `model_types` variable, this specifies
  the basic approach used to assemble the objects (rules, pairwise classifiers (+ maybe interaction model),
  classifiers trained through the interaction model).
  - `indices`: Specifies the indices to segment and plot for the data set.
  - `clf_type`: Has to be one of the classifier types present in the `clf_types` variable and specifies
  the type of the pairwise classifier used to predict a link for a pair of surfaces. For
  `model_type="Rules"` this parameter is irrelevant, for `model_type="CRF"` only the two differentiable
  classifiers can be used.
  - `use_CRF`: This parameter specifies whether the interaction model shall be used. This is only
  relevant for `model_type="Pairs"`.
  - `do_post_processing`: This parameter specifies whether the post-processing procedure shall be used.
- `complete_segmentation()`: This method creates a segmentation for a given image from the data set and plots it.
The parameters are similar to the parameters for the `detect_objects_for_dataset()` function,
with the `index` parameter specifying the index of the image from the data set that is supposed to be segmented.
- `def evaluate_models()`: This method allows calculating the AvgIoU score for the given model specified
by the parameters. These parameters are similar to the `detect_objects_for_dataset()` function.

Sometimes the data sets that are used to train the models need to be recreated. This can have different reasons:
- For the data set used to train the pixel similarity function, the reason could be that a different data set
is wanted because the data set creation is random and might influence the performance of the classifier.
- For the data sets used to train the pairwise surface link classifiers (alone or through the CRF)
the reason for recreating these training sets could be that the model for detecting surfaces has been
retrained. Then, the classifiers need to be retrained as well on these new surfaces.

The retraining of the data sets has to be done manually in this case, which can be done using the following methods
from the `main.py` file:
- `recreate_pixel_similarity_data_set()`
- `recreate_surface_link_data_set()`
- `recreate_crf_data_set()`



Note: The time required to create a segmentation of an image can differ from the time reported in the thesis,
because the segmentation of the first image always requires additional time. This is due to delays in
the initialization of the tensorflow models, and because the numba package is used which creates C
code from python code, which is done during the first segmentation as well and thus takes additional time.

