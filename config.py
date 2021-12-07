import math

height = 480
width = 640

viewing_angle_x = 62.0 / 180 * math.pi
viewing_angle_y = 48.6 / 180 * math.pi

test_indices = [2, 20, 22, 27, 30, 33, 37, 62, 79, 90, 92, 98, 101, 105, 107]
train_indices = [x for x in range(111) if x not in test_indices]

# The number of pairwise features for each pair of surfaces that are used as input for the pairwise classifiers.
num_pairwise_features = 14