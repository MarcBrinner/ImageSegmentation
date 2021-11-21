import plot_image
from find_planes import *

def apply_CRF(data, CRF):
    size_x, size_y = int(width / div_x), int(height / div_y)
    grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = np.stack([grid[1], grid[0]], axis=-1)

    features = extract_features(data["depth"], data["rgb"], data["angles"], grid)
    unary_potentials, initial_Q, prob = get_unary_potentials_and_initial_probabilities({"patches": data["final_surfaces"], "num_surfaces": int(np.max(data["final_surfaces"]) + 1)})

    inputs = get_inputs(features, unary_potentials, initial_Q, 7, div_x, div_y, size_x, size_y)
    out = CRF.predict(inputs, batch_size=1)
    Q = assemble_outputs(out, div_x, div_y, size_x, size_y, height, width)
    Q[data["depth"] == 0] = prob
    #plot_image.plot_surfaces(data["final_surfaces"])
    data["final_surfaces"] = np.argmax(Q, axis=-1)
    data["final_surfaces"][data["final_surfaces"] == data["num_surfaces"]] = 0
    #plot_image.plot_surfaces(data["final_surfaces"])
    return data

def get_postprocessing_model(kernel_size=7):
    size_x, size_y = int(width / div_x), int(height / div_y)
    p = load_Gauss_parameters()
    #p[3][0][0] = p[3][0][0] + 3
    CRF = conv_crf_Gauss(*p, kernel_size, size_y, size_x)
    #return lambda x: x
    return lambda x: apply_CRF(x, CRF)


