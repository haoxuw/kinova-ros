import argparse
def argumentParser():
    name = "final"
    output_dir = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/" + name + "_results/"
    input_dir = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/tracked_results/"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', type=str, default= input_dir, help='')
    parser.add_argument('--output_dir', type=str, default= output_dir, help='Path to output directory. Might overwrite.')
    parser.add_argument('--epochs', type=int, default=64, help='')
    parser.add_argument('--batch', type=int, default=16, help='')
    parser.add_argument('--steps_per_epoch', type=int, default=1024, help='')

    parser.add_argument('--max_size', type=int, default=-1, help='')

    parser.add_argument('--itera', type=int, default=0, help='')
    parser.add_argument('--re_train', action="store_true")
    parser.add_argument('--view', action="store_true")

    args = parser.parse_args()

    args.__TEST_SIZE__ = 1000
    args.__TEST_RATIO__ = 0.1

    args.__DECO_MODEL_NAME__ = "hw_deco"
    args.__DISC_MODEL_NAME__ = "hw_disc"
    args.__GENE_MODEL_NAME__ = "hw_gene"
    args.__AUTO_MODEL_NAME__ = "hw_auto"
    args.__DECO_FOLDER__ = "/NN_model_deco/"
    args.__DISC_FOLDER__ = "/NN_model_disc/"
    args.__GENE_FOLDER__ = "/NN_model_gene/"
    args.__AUTO_FOLDER__ = "/NN_model_auto/"

    args.__FIG_FOLDER__ = "/NN_model_saved_figures/"


    return args

args = argumentParser()

