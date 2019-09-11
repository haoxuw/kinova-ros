from learn_discriminator import *
from process_scripts import *

import sys, datetime
#import tensorflow as tf
from tensorflow.keras.layers import *

__DISC_MODEL_NAME__ = "hw_disc"
__GENE_MODEL_NAME__ = "hw_gene"
__AUTO_MODEL_NAME__ = "hw_auto"
__DECODER_NAME__ = "hw_decoder"
__DISC_FOLDER__ = "/disc/"
__GENE_FOLDER__ = "/gene/"
__AUTO_FOLDER__ = "/auto/"

def HW_autoencoder(input_shape):
    __NAME__ = __AUTO_MODEL_NAME__
    __NAME_PREVIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 7

    __SEED_TRANS_NUM__ = 5

    __SHAPE__ = [32, 64, 128, 256, 512, 512, 512, 512, 1024, 2048]


    conv_shape = __SHAPE__
    deconv_shape = __SHAPE__
    filter_size = __FILTER_SIZE__



    def add_repetition_unit_v1(x, seed, filters, j):
        x = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv_%d_num_%d' %(filters, j) )(x)
        return x

    def add_repetition_unit_v2(x, seed, filters, j):
        ori_x = x

        base_layer = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv1_%d_num_%d' %(filters, j) )
        x1 = base_layer(x)

        x21 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREVIX__+ 'deconv21_%d_num_%d' %(filters, j) )(x)
        x2 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv2_%d_num_%d' %(filters, j) )(x)

        x31 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREVIX__+ 'deconv31_%d_num_%d' %(filters, j) )(x)
        x32 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREVIX__+ 'deconv32_%d_num_%d' %(filters, j) )(x31)
        x3 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv3_%d_num_%d' %(filters, j) )(x32)

        s = seed
        (b_size, x_size, y_size, c_size) = base_layer.output_shape
        s51 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense51_%d_num_%d" %(filters, j) )(s)
        s52 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense52_%d_num_%d" %(filters, j) )(s51)
        s53 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense53_%d_num_%d" %(filters, j) )(s52)
        s54 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense54_%d_num_%d" %(filters, j) )(s53)
        s5 = keras.layers.Dense(x_size * c_size, activation=activation, name = __NAME_PREVIX__ + "dense5_%d_num_%d" %(filters, j) )(s54)
        s5 = tf.keras.layers.Reshape([x_size, y_size, c_size])(s5)

        x = keras.layers.Add(name = __NAME_PREVIX__+ 'add_num_%d' %(j))([x1, x2, x3, s5])
        return x






    ori_input = keras.Input(shape=input_shape[1:], name = __NAME_PREVIX__ + "input")

    x = ori_input

    stride = 1
    padding = 0
    activation = 'relu'

    seed_trans_num = __SEED_TRANS_NUM__

    last_layer_x = input_shape[1]
    last_index = 0
    for i in range(len(conv_shape)):
        current_layer_x = (last_layer_x - filter_size + 2 * padding) / stride + 1
        #print last_layer_x
        if current_layer_x < 2:
            break
        filters = conv_shape[i]
        x = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREVIX__+ 'conv1d_%d_num_%d' %(last_layer_x, i) )(x)
        last_layer_x = current_layer_x
        last_index = i
        print i
        print last_layer_x
        print 

    x = keras.layers.Conv1D(6, last_layer_x, activation=activation, name = __NAME_PREVIX__+ 'conv1d_bottlenek' )(x)

    x = tf.keras.layers.Reshape((1,6), name = __DECODER_NAME__ + "_entry")(x)

    seed = x

    seed2d = tf.keras.layers.Reshape((1,1,6), name = __NAME_PREVIX__ + "seed2d")(seed)

    start = seed2d
    for i in range(seed_trans_num):
        start = keras.layers.Dense(6, activation=activation, name = __NAME_PREVIX__ + "dense_start_num_%d" %(i) )(start)
    
    x = start
    for i in range(seed_trans_num):
        x = keras.layers.Dense(6, activation=activation, name = __NAME_PREVIX__ + "dense_seed_num_%d" %(i) )(x)
    
    x = keras.layers.Conv2DTranspose(6, (last_layer_x, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv_first' )(x)

    transeconv_shape = range(last_index)
    transeconv_shape.reverse()
    print "!"
    for j in transeconv_shape:
        filters = conv_shape[j]
        x = add_repetition_unit_v1(x, seed2d, filters, j)


    x = keras.layers.Conv2DTranspose(6, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv_last_layer' )(x)

    x = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name = __NAME_PREVIX__ + "cropping")(x)

    x = tf.keras.layers.concatenate( [ start , x], axis = 1)

    x = tf.keras.layers.Reshape((64,6), name = __DECODER_NAME__ + "_exit")(x)

    exit = x

    model = keras.Model(inputs=ori_input, outputs=exit , name= __NAME__)

    model.compile(optimizer = 'adam',
                  loss = 'MSE',
                  )

    print model.summary()

    return model

def HW_Gen_CONV(decoder_pack, discriminator_pack):
    __NAME__ = __GENE_MODEL_NAME__
    __NAME_PREVIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 9

    __DECONV_SHAPE__ = [32, 64, 128, 256, 512, 512, 1024, 2048]
    __DENSE_SHAPE__ = [32]

    seed = keras.Input(shape=(1,6), name = __NAME_PREVIX__ + "seed")
    x = seed

    [deco_entry, deco_exit, deco_model] = decoder_pack
    [disc_entry, disc_exit, disc_model] = discriminator_pack
    deco_model.trainable = True

    print deco_model.summary()

    for layer in deco_model.layers[:]:
        layer.trainable = True

        print layer.name

        if layer.name != deco_entry.name:
            deco_model._layers.pop(0)
        else:
            break

    print deco_model.summary()
    deco_model.compile(optimizer = 'adam', loss = "MAE")

    deco_model_sub = keras.Model(inputs=deco_model.layers[0].input, outputs=deco_model.layers[-1].output, name= __GENE_MODEL_NAME__ + "_deco_model_sub")

    decoder_entry = seed
    print deco_entry
    print dir(deco_entry)
    deco_entry.input(seed)
    print decoder_exit

    print deco_model.summary()

    decoder_sub_graph = keras.Model(inputs=seed, outputs=deco_exit.output, name= __GENE_MODEL_NAME__ + "_decoder_sub_graph")
    print decoder_sub_graph.summary()
    sys.exit()

    #print sub_deco.summary()
    sys.exit()





    x = sub_deco(x)

    sub_disc = keras.Model(inputs=disc_entry, outputs=disc_exit , name= __GENE_MODEL_NAME__ + "_sub_disc")

    for layer in sub_disc.layers:
        layer.trainable = False

    x = sub_disc(x)

    model = keras.Model(inputs=seed, outputs=x , name= __GENE_MODEL_NAME__)
    
    model.compile(optimizer = 'adam', loss = "MAE")

    print model.summary()

    return model

def create_generator(decoder_pack, discriminator_pack):
    print "\n\n-------- create generator  ----------\n\n"
    
    model = HW_Gen_CONV(decoder_pack = decoder_pack, discriminator_pack = discriminator_pack)
    
    return model

def load_model_pack(path, model_name, trainable = False, entry = "entry", exit = "exit"):
    model = tf.keras.models.load_model(path + "/" + model_name + ".h5")
    model.trainable = trainable

    model.compile(optimizer='adam',
                  loss='MAE',
    )

    print "Loaded model %s" % model.name

    discri = [None, None, model]
    for layer in model.layers:
        if entry in layer.name:
            discri[0] = layer
        elif exit in layer.name:
            discri[1] = layer

    #print model.summary()

    return discri

def gen_seeds(size = 10000):
    seeds = np.random.rand(size, 6)

    seeds [:,4] = 0
    seeds /= 2

    return seeds

def create_and_fit_autoencoder(affine_train, affine_test, path):
    auto_model = HW_autoencoder(affine_train.shape)

    affine_data_train = [affine_train, affine_train]
    affine_data_test = [affine_test, affine_test]

    fit_and_save_model(auto_model, affine_data_train, affine_data_test, path, epoch = epoch, visualize = False)

    return auto_model

def create_and_fit_generator(decoder_pack, discr_pack, path, size = 10000):

    gen_model = create_generator( decoder_pack = decoder_pack, discriminator_pack = discr_pack )
    return
    seeds = gen_seeds(size)
    losses = np.zeros(size)
    random_data = [seeds , losses]

    sampled_data = random_data

    gen_model = fit_and_save_model(gen_model, sampled_data, path)

    gen_predicted_data = predict_scripts(gen_model, seeds)

    return gen_model, gen_predicted_data


def predict_scripts(model, seeds):
    [losses, scripts] = model.predict(seeds)
    losses += 0.8

    #losses = losses.reshape(-1)

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

    print sampled_data1[0].shape
    print sampled_data2[0].shape
    print sampled_data1[1].shape
    print sampled_data2[1].shape
    sample = [
        np.concatenate([sampled_data1[0], sampled_data2[0]], axis = 0),
        np.concatenate([sampled_data1[1], sampled_data2[1]], axis = 0)
    ]
    return sample


def run_iteration(demo_data, iteration = 0, size = 10000, save_samples = "/saved_figs/"):

    return 
    #print demo_data[0].shape
    #print gen_predicted_data[0].shape
    #print gen_predicted_data[1].shape
    if save_samples:
        path = output_path + save_samples
        os.system('mkdir -p ' + path)
        for pred_index in range(0, size, size / 10):
            script = gen_predicted_data[0][pred_index]
            dist = gen_predicted_data[1][pred_index]
            pred_script = np.copy(script).reshape([64,6])

            visualize_script(pred_script, filename = path + "/iter_" + str(iteration) + "_fig_" + str(pred_index) + ".png", dist = dist)

    print "\n\n-------- load_discriminator, trainable  ----------\n\n"
    discr_model = load_model_pack(output_path + __DISC_FOLDER__, __DISC_MODEL_NAME__, trainable = True)[2]

    sample = merge_data(gen_predicted_data, demo_data, size = size)

    print "\n\n-------- re-train_discriminator with sampled gen+demo data ----------\n\n"
    fit_and_save_model(discr_model, sample, None, output_path)

    return gen_model

def main():
    size = 100#-1
    print "\n\n-------- loading train/test dataset  ----------\n\n"
    #test, train, affine_test, affine_train = load_np_data(output_path, max_size = size, visualize = False)

    print "\n\n-------- create_and_fit_autoencoder  ----------\n\n"
    #auto_model = create_and_fit_autoencoder (affine_train, affine_test, path = output_path + __AUTO_FOLDER__)

    print "\n\n-------- load_decoder from autoencoder  ----------\n\n"
    decoder_pack = load_model_pack (path = output_path + __AUTO_FOLDER__, model_name = __AUTO_MODEL_NAME__, entry = __DECODER_NAME__ + "_entry", exit = __DECODER_NAME__ + "_exit") 

    print "\n\n-------- load_discriminator  ----------\n\n"
    discr_pack = load_model_pack(path = output_path + __DISC_FOLDER__, model_name = __DISC_MODEL_NAME__)

    print "\n\n-------- create_and_fit_generator  ----------\n\n"
    gen_model, gen_predicted_data = create_and_fit_generator(decoder_pack = decoder_pack, discr_pack = discr_pack, path = output_path + __GENE_FOLDER__, size = size)

    print decoder
    print decoder[2].summary()
    return
    #y = train[1]
    #print " >>>>>>>>>>>>>>> y %r %r " % ( y.shape , y.mean(axis = 0))

    for itera in range(100):
        run_iteration(train, iteration = itera, size = size * (itera+1) )

    print gen_model.summary()
    
if __name__== "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
