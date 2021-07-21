import math

height = 480
width = 640
viewing_angle_x = 62.0 / 180 * math.pi
viewing_angle_y = 48.6 / 180 * math.pi
factor_x = math.tan(viewing_angle_x / 2) * 2 / width
factor_y = math.tan(viewing_angle_y / 2) * 2 / height