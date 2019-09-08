from learn_discriminator import *
from process_scripts import *

import sys, datetime
#import tensorflow as tf
from tensorflow.keras.layers import *

__DISC_MODEL_NAME__ = "hw_disc"

def HW_autoencoder(input_shape):
    __NAME__ = 'hw_au'
    __NAME_PREVIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 7

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
        x2 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv2_%d_num_%d' %(filters, j) )(x21)

        x31 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREVIX__+ 'deconv31_%d_num_%d' %(filters, j) )(x)
        x32 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREVIX__+ 'deconv32_%d_num_%d' %(filters, j) )(x31)
        x3 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv3_%d_num_%d' %(filters, j) )(x32)

        s = seed
        (b_size, x_size, y_size, c_size) = base_layer.output_shape
        s51 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense51_%d_num_%d" %(filters, j) )(s)
        s52 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense52_%d_num_%d" %(filters, j) )(s)
        s53 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense53_%d_num_%d" %(filters, j) )(s)
        s54 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREVIX__ + "dense54_%d_num_%d" %(filters, j) )(s)
        s5 = keras.layers.Dense(x_size * c_size, activation=activation, name = __NAME_PREVIX__ + "dense5_%d_num_%d" %(filters, j) )(s54)
        s5 = tf.keras.layers.Reshape([x_size, y_size, c_size])(s5)


        x = x1 + x2 + x3 + s5
        return x






    entry = keras.Input(shape=input_shape[1:], name = __NAME_PREVIX__ + "entry")

    x = entry

    stride = 1
    padding = 0
    activation = 'relu'

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

    x = tf.keras.layers.Reshape((1,6), name = __NAME_PREVIX__ + "seed")(x)

    seed = x

    seed2d = tf.keras.layers.Reshape((1,1,6), name = __NAME_PREVIX__ + "seed2d")(x)

    x = keras.layers.Conv2DTranspose(6, (last_layer_x, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv_first' )(seed2d)

    transeconv_shape = range(last_index)
    transeconv_shape.reverse()
    print "!"
    for j in transeconv_shape:
        filters = conv_shape[j]
        x = add_repetition_unit_v1(x, seed2d, filters, j)


    x = keras.layers.Conv2DTranspose(6, (filter_size, 1), activation=activation, name = __NAME_PREVIX__+ 'deconv_last_layer' )(x)

    x = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name = __NAME_PREVIX__ + "cropping")(x)

    x = tf.keras.layers.concatenate( [ seed2d , x], axis = 1)

    x = tf.keras.layers.Reshape((64,6), name = __NAME_PREVIX__ + "exit")(x)

    exit = x

    model = keras.Model(inputs=entry, outputs=exit , name= __NAME__)

    model.compile(optimizer = 'adam',
                  loss = 'MSE',
                  )

    print model.summary()

    return model

def HW_Gen_CONV(discriminator):
    __NAME__ = 'hw_gen'
    __NAME_PREVIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 9

    __DECONV_SHAPE__ = [32, 64, 128, 256, 512, 512, 1024, 2048]
    __DENSE_SHAPE__ = [32]

    seed = keras.Input(shape=(6), name = __NAME_PREVIX__ + "seed")
    x = seed

    [entry, exit, discriminator_model] = discriminator
    disc_input_shape = entry.input_shape[0][1:]
    #print disc_input_shape

    dense_shape = __DENSE_SHAPE__
    activation = 'relu'
    #activation = 'linear'
    #activation = 'sigmoid'

    for cnt,filters in enumerate(dense_shape):
        x = keras.layers.Dense(filters, activation=activation, name = __NAME_PREVIX__ + "dense_" + str(cnt) )(x)

    x = tf.keras.layers.Reshape([1, 1, dense_shape[-1]])(x)

    deconv_shape = __DECONV_SHAPE__

    kernel_size = (__FILTER_SIZE__ , 1)

    cnt = 1
    for cnt, filters in enumerate(deconv_shape):
        layer = keras.layers.Conv2DTranspose ( filters, kernel_size, activation=activation,
                                               name = __NAME_PREVIX__ + "transpseconv_" + str(cnt) )
        x = layer(x)

    x = keras.layers.Cropping2D(cropping=((1, 1), (0, 0)), name = __NAME_PREVIX__ + "cropping")(x)

    cnt = 1
    last_layer_channel = deconv_shape[-1] / 2
    while last_layer_channel > 6:
        layer = keras.layers.Conv2D (last_layer_channel, [1,1], activation=activation,
                                     name = __NAME_PREVIX__ + "1X1_conv_" + str(cnt) )
        x = layer(x)
        cnt += 1

        last_layer_channel /= 2

    last_activation = 'sigmoid'
    x = keras.layers.Conv2D (6, [1,1],
                             activation = last_activation,
                             name = __NAME_PREVIX__ + "1X1_conv_last" )(x)


    seed_2d = tf.keras.layers.Reshape([1, 1, 6])(seed)
    x = tf.keras.layers.concatenate( [ seed_2d , x], axis = 1)

    x = tf.keras.layers.Reshape(disc_input_shape, name = __NAME_PREVIX__ + "script")(x)

    script_tensor = x

    '''
    pos = script_tensor[:,:,0:3]

    pos = tf.keras.layers.Reshape([64, 3])(pos)

    avg_pos = tf.keras.layers.AveragePooling1D (pool_size = 64)(pos)
    '''

    discriminator_loss = discriminator_model(script_tensor)

    #avg_pos_loss = keras.layers.Subtract(name = __NAME_PREVIX__ + "avg_pos_loss_layer") ([seed, avg_pos])

    #loss = (discriminator_loss + avg_pos_loss)
    loss = discriminator_loss
    #loss = avg_pos_loss

    #loss = tf.keras.layers.Multiply(name = __NAME_PREVIX__ + "loss_layer_output") ([discriminator_loss, avg_pos_loss])
    loss = tf.keras.layers.Multiply(name = __NAME_PREVIX__ + "loss_layer_output") ([loss, loss])

    model = keras.Model(inputs=seed, outputs=[loss, script_tensor] , name= __NAME__)

    for layer in model.layers:
        layer.trainable = True

    for layer in discriminator_model.layers:
        layer.trainable = False

    model.compile(optimizer = 'adam',
                  loss = {
                      #"tf_op_layer_mul" : 'MSE'
                      __NAME_PREVIX__ + 'loss_layer_output' : 'MAE',
                      },
                  loss_weights = {
                      #"tf_op_layer_mul" : 1.
                      __NAME_PREVIX__ + 'loss_layer_output' : 1.,
                      },
                  )

    print model.summary()

    return model

def create_generator(discriminator):
    model = HW_Gen_CONV(discriminator)
    
    return model

def load_discriminator(path, model_name, trainable = False):
    model = tf.keras.models.load_model(path + "/" + model_name + ".h5")
    model.trainable = trainable

    model.compile(optimizer='adam',
                  loss='MAE',
    )

    print "Loaded model %s" % model.name

    discri = [None, None, model]
    for layer in model.layers:
        if "entry" in layer.name:
            discri[0] = layer
        elif "exit" in layer.name:
            discri[1] = layer

    #print model.summary()

    return discri

def gen_seeds(size = 10000):
    seeds = np.random.rand(size, 6)

    seeds [:,4] = 0
    seeds /= 2

    return seeds

def create_and_fit_autoencoder(affine_train, affine_test, size = 10000):
    auto_model = HW_autoencoder(affine_train.shape)

    affine_data_train = [affine_train, affine_train]
    affine_data_test = [affine_test, affine_test]

    for i in range(0, 100, 1):
        x = affine_test[i]
        #visualize_script(x, dist = 0.031)

    fit_model(auto_model, affine_data_train, affine_data_test, output_path + "/auto/", epoch = epoch*7)

    return auto_model

def create_and_fit_generator(discr, demo_data, size = 10000):

    discr_model = discr[2]

    gen_model = create_generator( discr )

    seeds = gen_seeds(size)
    losses = np.zeros(size)
    random_data = [seeds , losses]

    #sampled_data = merge_data(random_data, demo_data, size = size)
    sampled_data = random_data

    gen_model = fit_model(gen_model, sampled_data, None, output_path + "/generator/")

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
    disc_name = __DISC_MODEL_NAME__

    print "\n\n-------- load_discriminator  ----------\n\n"
    discr = load_discriminator(output_path, disc_name)

    print "\n\n-------- create_and_fit_generator  ----------\n\n"
    gen_model, gen_predicted_data = create_and_fit_generator(discr = discr, demo_data = demo_data, size = size)
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
    discr_model = load_discriminator(output_path, disc_name, True)[2]

    sample = merge_data(gen_predicted_data, demo_data, size = size)

    print "\n\n-------- re-train_discriminator with sampled gen+demo data ----------\n\n"
    fit_model(discr_model, sample, None, output_path)

    return gen_model

def main():
    test, train, affine_test, affine_train = load_np_data(output_path, max_size = -1, visualize = True)
    size = 10000

    auto_model = create_and_fit_autoencoder (affine_train, affine_test, size)
    return
    #y = train[1]
    #print " >>>>>>>>>>>>>>> y %r %r " % ( y.shape , y.mean(axis = 0))

    for itera in range(100):
        run_iteration(train, iteration = itera, size = size * (itera+1) )

    print gen_model.summary()
    
if __name__== "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
