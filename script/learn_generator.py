from learn_discriminator import *
from process_scripts import *

import sys, datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__DISC_MODEL_NAME__ = "hw_disc"
__GENE_MODEL_NAME__ = "hw_gene"
__DECO_MODEL_NAME__ = "hw_deco"
__DISC_FOLDER__ = "/disc/"
__GENE_FOLDER__ = "/gene/"
__DECO_FOLDER__ = "/deco/"

def HW_decoder(input_shape):
    __NAME__ = __DECO_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 7

    __SEED_TRANS_NUM__ = 5

    __SHAPE__ = [32, 64, 128, 256, 512, 512, 512, 512, 1024, 2048]


    conv_shape = __SHAPE__
    deconv_shape = __SHAPE__
    filter_size = __FILTER_SIZE__



    def add_repetition_unit_v1(x, seed, filters, j):
        x = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv_%d_num_%d' %(filters, j) )(x)
        return x

    def add_repetition_unit_v2(x, seed, filters, j):
        ori_x = x

        base_layer = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv1_%d_num_%d' %(filters, j) )
        x1 = base_layer(x)

        x21 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv21_%d_num_%d' %(filters, j) )(x)
        x2 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv2_%d_num_%d' %(filters, j) )(x)

        x31 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv31_%d_num_%d' %(filters, j) )(x)
        x32 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv32_%d_num_%d' %(filters, j) )(x31)
        x3 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv3_%d_num_%d' %(filters, j) )(x32)

        s = seed
        (b_size, x_size, y_size, c_size) = base_layer.output_shape
        s51 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREFIX__ + "dense51_%d_num_%d" %(filters, j) )(s)
        s52 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREFIX__ + "dense52_%d_num_%d" %(filters, j) )(s51)
        s53 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREFIX__ + "dense53_%d_num_%d" %(filters, j) )(s52)
        s54 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREFIX__ + "dense54_%d_num_%d" %(filters, j) )(s53)
        s5 = keras.layers.Dense(x_size * c_size, activation=activation, name = __NAME_PREFIX__ + "dense5_%d_num_%d" %(filters, j) )(s54)
        s5 = keras.layers.Reshape([x_size, y_size, c_size])(s5)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' %(j))([x1, x2, x3, s5])
        return x






    seed = keras.layers.Input(shape=input_shape[1:], name = __NAME_PREFIX__ + "entry")

    stride = 1
    padding = 0
    activation = 'relu'

    seed_trans_num = __SEED_TRANS_NUM__

    last_layer_x = 64
    last_index = 0
    for i in range(len(conv_shape)):
        current_layer_x = (last_layer_x - filter_size + 2 * padding) / stride + 1
        #print last_layer_x
        if current_layer_x < 2:
            break
        filters = conv_shape[i]
        last_layer_x = current_layer_x
        last_index = i

    seed2d = keras.layers.Reshape((1,1,6), name = __NAME_PREFIX__ + "seed2d")(seed)

    start = seed2d
    x = start
    for i in range(seed_trans_num):
        x = keras.layers.Dense(6, activation=activation, name = __NAME_PREFIX__ + "dense_start_num_%d" %(i) )(x)

    x = keras.layers.Conv2DTranspose(6, (last_layer_x, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv_first' )(x)

    transeconv_shape = range(last_index)
    transeconv_shape.reverse()
    for j in transeconv_shape:
        filters = conv_shape[j]
        x = add_repetition_unit_v2(x, seed2d, filters, j)


    x = keras.layers.Conv2DTranspose(6, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv_last_layer' )(x)

    x = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name = __NAME_PREFIX__ + "cropping")(x)

    x = keras.layers.Concatenate(axis = 1)([ start , x])

    x = keras.layers.Reshape((64,6), name = __DECO_MODEL_NAME__ + "_exit")(x)

    exit = x

    model = keras.Model(inputs=seed, outputs=exit , name= __NAME__)

    model.compile(optimizer = 'adam',
                  loss = 'MSE',
                  )

    print model.summary()

    return model

def HW_Gene_Minimal_GAN(decoder_pack, discriminator_pack):
    __NAME__ = __GENE_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 9

    __DECONV_SHAPE__ = [32, 64, 128, 256, 512, 512, 1024, 2048]
    __DENSE_SHAPE__ = [32]

    seed = keras.layers.Input(shape=(1,6), name = __NAME_PREFIX__ + "entry")
    x = seed

    [deco_entry, deco_exit, deco_model] = decoder_pack
    [disc_entry, disc_exit, disc_model] = discriminator_pack

    deco_model.trainable = True
    disc_model.trainable = False

    x = deco_model(x)

    script = x

    x = disc_model(x)
    disc_loss = keras.layers.Reshape([1], name = __NAME_PREFIX__ + "exit" )(x)

    gene_model = keras.Model(inputs=seed, outputs=[disc_loss, script] , name= __GENE_MODEL_NAME__)

    return gene_model

def find_entry_exit(model, entry = "entry", exit = "exit"):
    model_pack = [None, None, model]
    for layer in model.layers:
        if entry in layer.name:
            model_pack[0] = layer
        elif exit in layer.name:
            model_pack[1] = layer

    return model_pack


def config_generator(gene_model, deco_trainable, disc_trainable):
    for layer in gene_model.layers:
        if __DECO_MODEL_NAME__ == layer.name:
            print "-------- configure %s : %r  ----------" % (layer.name, deco_trainable)
            layer.trainable = deco_trainable
        elif __DISC_MODEL_NAME__ == layer.name:
            print "-------- configure %s : %r  ----------\n\n" % (layer.name, disc_trainable)
            layer.trainable = disc_trainable
    gene_model = recompile(gene_model)
    return gene_model
    
def recompile(model):
    print "\n\n-------- recompiled %r ----------\n\n" % model.name

    entry, exit, model = find_entry_exit(model)
        
    model.compile(optimizer = 'adam',
                       loss = {
                           exit.name : 'MSE',
                       },
                       loss_weights = {
                           exit.name : 1
                       })

    print model.summary()

    return model

def create_generator(decoder_pack, discriminator_pack):
    print "\n\n-------- create generator  ----------\n\n"
    
    gene_model = HW_Gene_Minimal_GAN(decoder_pack = decoder_pack, discriminator_pack = discriminator_pack)
    gene_model = recompile (gene_model)
    return gene_model


def load_model_pack(path, model_name, trainable = False, entry = "entry", exit = "exit"):
    model = keras.models.load_model(path + "/" + model_name + ".h5")
    if trainable != None:
        model.trainable = trainable

    model.compile(optimizer='adam',
                  loss='MSE',
    )

    print "Loaded model %s" % model.name

    discri = find_entry_exit(model, entry, exit)
    #print model.summary()

    return discri

def gen_seeds(size = 10000):
    seeds = np.random.rand(size, 6)

    seeds [:,4] = 0
    seeds /= 2

    return seeds.reshape([-1,1,6])

def create_and_fit_decoder(affine_train, affine_test, path):
    deco_model = HW_decoder([None,1,affine_train.shape[-1]])

    affine_data_train = [affine_train[:,0:1,:], affine_train]
    affine_data_test = [affine_test[:,0:1,:], affine_test]

    fit_and_save_model(deco_model, affine_data_train, affine_data_test, path, epoch = epoch, visualize = False)

    return deco_model

def fit_generator_decoder(gene_model, demo_data, path, size = 10000):
    print "\n\n-------- re-train decoder with sampled demo+random seed ----------\n\n"

    seeds = gen_seeds(size)
    losses = np.zeros(size).reshape(-1,1)

    random_data = [seeds , losses]
    affine_data = [ demo_data[:,0:1,:], np.zeros(demo_data.shape[0]).reshape(-1,1) ]

    sampled_data = merge_data(random_data, affine_data, size, 0.5)

    config_generator (gene_model, deco_trainable = True, disc_trainable = False)
    gene_model = fit_and_save_model(gene_model, train = sampled_data, test = None, model_output_path = path)

    gen_predicted_data = predict_scripts(gene_model, seeds)

    return gene_model, gen_predicted_data

def fit_generator_discriminator(gene_model, demo_data, gen_predicted_data, path, size = 10000):
    print "\n\n-------- re-train discriminator with sampled demo+gene_predicted seed ----------\n\n"

    gen_predicted_data[0] = gen_predicted_data[0][:,0:1,:]
    gen_predicted_data[1] += 0.8

    gen_predicted_data_loss = gen_predicted_data[1]

    affine_data = [ demo_data[:,0:1,:], np.zeros(demo_data.shape[0]).reshape(-1,1) ]

    sampled_data = merge_data(affine_data, gen_predicted_data, size = size, ratio = 0.5)

    config_generator (gene_model, deco_trainable = False, disc_trainable = True)
    gene_model = fit_and_save_model(gene_model, train = sampled_data, test = None, model_output_path = path)

    gen_predicted_data = predict_scripts(gene_model, gen_predicted_data[0])

    return gene_model, gen_predicted_data

def predict_scripts(model, seeds):
    [losses, scripts] = model.predict(seeds)

    scripts = scripts.reshape(seeds.shape[0], 64, 6)

    data = [scripts, losses]
    return data

def sample_from(data, size):
    max_size = data[0].shape[0]
    sample_x = data[0][np.random.choice(max_size, size)]
    sample_y = data[1][np.random.choice(max_size, size)]
    sample = [sample_x, sample_y]
    return sample

def merge_data(data1, data2, size = 10000, ratio = 0.5):
    sampled_data1 = sample_from(data1, int(size * ratio))
    sampled_data2 = sample_from(data2, size - int(size * ratio) )

    if False:
        print sampled_data1[0].shape
        print sampled_data2[0].shape
        print sampled_data1[1].shape
        print sampled_data2[1].shape

    sample = [
        np.concatenate([sampled_data1[0], sampled_data2[0]], axis = 0),
        np.concatenate([sampled_data1[1], sampled_data2[1]], axis = 0)
    ]
    return sample

def save_sample_figures(data, save_path, size = 10, prefix = "sample_fig", max_fig_size = 20):
    if save_path is not None:
        path = output_path + save_path
        os.system('mkdir -p ' + path)

        total_size = data[0].shape[0]
        for pred_index in range(0, total_size, total_size / size):
            script = data[0][pred_index]
            dist = data[1][pred_index]
            pred_script = np.copy(script).reshape([64,6])
            max_fig_size -= 1
            if (max_fig_size < 1):
                break

            visualize_script(pred_script, filename = path + "/" + prefix + "_fig_" + str(pred_index) + ".png", dist = dist)


def run_iteration(gene_model, affine_data, iteration = 0, size = 10000, path = __GENE_FOLDER__, save_samples = "/saved_figs/"):
    print "\n\n-------- Running Iteration %d  ----------\n\n" % iteration
    #need to load model every iteration to workaround what seems to be a tensorflow mem leak
    keras.backend.clear_session()
    gene_model = load_model_pack(output_path + __GENE_FOLDER__, __GENE_MODEL_NAME__, trainable = None)[2]

    print "\n\n-------- acquired generator  ----------\n\n"
    if iteration == 0:
        print gene_model.summary()

    gene_model, gen_predicted_data = fit_generator_decoder(gene_model = gene_model, demo_data = affine_data, path = path, size = size)

    prefix = "posdeco_iter_" + str(iteration)
    save_sample_figures(gen_predicted_data, save_samples, size, prefix)

    gene_model, gen_predicted_data = fit_generator_discriminator(gene_model = gene_model, demo_data = affine_data, gen_predicted_data = gen_predicted_data, path = path, size = size)

    prefix = "posdisc_iter_" + str(iteration)
    save_sample_figures(gen_predicted_data, save_samples, size, prefix)

    return gene_model

def main():
    size = 100000#-1

    re_train = False
    #re_train = True
    
    print "\n\n-------- loading train/test dataset  ----------\n\n"
    test, train, affine_test, affine_train = load_np_data(output_path, max_size = size, visualize = False)
    print "\ttrain size %d, affine_train size %d" % (train[0].shape[0], affine_train[0].shape[0])

    if re_train:
        print "\n\n-------- create_and_fit_discriminator  ----------\n\n"
        create_and_fit_discriminator(train = train, test = test, path = output_path + __DISC_FOLDER__)

        print "\n\n-------- create_and_fit_decoder  ----------\n\n"
        deco_model = create_and_fit_decoder (affine_train = affine_train, affine_test = affine_test, path = output_path + __DECO_FOLDER__)

        print "\n\n-------- load_decoder from decoder  ----------\n\n"
        decoder_pack = load_model_pack (path = output_path + __DECO_FOLDER__, model_name = __DECO_MODEL_NAME__, entry = __DECO_MODEL_NAME__ + "_entry", exit = __DECO_MODEL_NAME__ + "_exit") 

        print "\n\n-------- load_discriminator  ----------\n\n"
        discr_pack = load_model_pack(path = output_path + __DISC_FOLDER__, model_name = __DISC_MODEL_NAME__)

        print "\n\n-------- create_generator  ----------\n\n"
        gene_model = create_generator( decoder_pack = decoder_pack, discriminator_pack = discr_pack )
    else:
        print "\n\n-------- load_generator  ----------\n\n"
        gene_model = load_model_pack(output_path + __GENE_FOLDER__, __GENE_MODEL_NAME__, trainable = None)[2]



    total_iter = 256
    for itera in range(total_iter):
        gene_model = run_iteration(gene_model = gene_model, affine_data = affine_train, iteration = itera, path = output_path + __GENE_FOLDER__, size = size / total_iter * (itera+1) + 10 )

    return gene_model
    
if __name__== "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
