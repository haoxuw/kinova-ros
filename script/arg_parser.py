import argparse
def argumentParser():
    name = "pick_place"
    name = "pouring"
    output_dir = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/" + name + "_results/"
    input_dir = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/tracked_" + name + "/"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default= name, help='')
    parser.add_argument('--input_dir', type=str, default= input_dir, help='')
    parser.add_argument('--output_dir', type=str, default= output_dir, help='Path to output directory. Might overwrite.')
    parser.add_argument('--epochs', type=int, default=64, help='')
    parser.add_argument('--gen_steps', type=int, default=1, help='')
    parser.add_argument('--dis_steps', type=int, default=1, help='')
    parser.add_argument('--batch', type=int, default=64, help='')
    parser.add_argument('--steps_per_epoch', type=int, default=1024, help='')

    parser.add_argument('--max_size', type=int, default=100000, help='')

    parser.add_argument('--itera', type=int, default=0, help='')
    parser.add_argument('--init_gan', action="store_true")
    parser.add_argument('--train_gan', action="store_true")
    parser.add_argument('--state_to_action', action="store_true")
    parser.add_argument('--state_history', type=int, default=5, help='')
    parser.add_argument('--train_bc', action="store_true")
    parser.add_argument('--run_bc', action="store_true")
    parser.add_argument('--view', action="store_true")

    parser.add_argument('--save_fig_num', type=int, default=11, help='')
    parser.add_argument('--save_fig_folder', type=str, default="/NN_model_saved_figures/", help='')
    parser.add_argument('--save_log_name', type=str, default="train_log", help='')

    args = parser.parse_args()

    args.__TEST_SIZE__ = 1000
    args.__TEST_RATIO__ = 0.1

    args.__DECO_MODEL_NAME__ = "hw_Generator"
    args.__DISC_MODEL_NAME__ = "hw_Discriminator"
    args.__GAN_MODEL_NAME__  = "hw_GAN"
    args.__AUTO_MODEL_NAME__ = "hw_auto"
    args.__BC_MODEL_NAME__ = "hw_bc"
    args.__DECO_FOLDER__ = "/NN_model_gene/"
    args.__DISC_FOLDER__ = "/NN_model_disc/"
    args.__GAN_FOLDER__  = "/NN_model_gan/"
    args.__AUTO_FOLDER__ = "/NN_model_auto/"
    args.__BC_FOLDER__ = "/NN_model_bc/"

    args.__FIG_FOLDER__ = "/NN_model_saved_figures/"

    args.__MAX_ACCU_DATA_SIZE__ = 1000000

    return args

args = argumentParser()

