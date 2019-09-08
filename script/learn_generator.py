from learn_discriminator import *
from process_scripts import *

import sys, datetime
#import tensorflow as tf
from tensorflow.keras.layers import *

__DISC_MODEL_NAME__ = "hw_disc"

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

    model.compile(optimizer = 'adam',
                  loss = {
                      #"tf_op_layer_mul" : 'MSE'
                      __NAME_PREVIX__ + 'loss_layer_output' : 'MSE',
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

def create_and_fit_generator(discr, demo_data, size = 10000):

    discr_model = discr[2]

    gen_model = create_generator( discr )

    seeds = gen_seeds(size * 2)
    losses = np.zeros(size * 2)
    random_data = [seeds , losses]
    print seeds.shape
    print losses.shape
    print demo_data[0].shape
    print demo_data[1].shape

    #sampled_data = merge_data(random_data, demo_data, size = size)
    #todo figure out a way around fanishing G
    sampled_data = random_data

    gen_model = fit_model(gen_model, sampled_data, None, output_path + "/generator/")

    gen_predicted_data = predict_scripts(gen_model, seeds)

    return gen_model, gen_predicted_data


def predict_scripts(model, seeds):
    [losses, scripts] = model.predict(seeds)
    losses += 0.9

    #losses = losses.reshape(-1)

    scripts = scripts.reshape(-1, 64, 6)

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
            pred_script = add_time_axis(pred_script)

            visualize_script(pred_script, filename = path + "/iter_" + str(iteration) + "_fig_" + str(pred_index) + ".png", dist = dist)

    print "\n\n-------- load_discriminator, trainable  ----------\n\n"
    discr_model = load_discriminator(output_path, disc_name, True)[2]

    sample = merge_data(gen_predicted_data, demo_data, size = size)

    print "\n\n-------- re-train_discriminator with sampled gen+demo data ----------\n\n"
    fit_model(discr_model, sample, None, output_path)

    return gen_model

def main():
    test,train = load_np_data(output_path, max_size = -1)
    size = 10000

    #y = train[1]
    #print " >>>>>>>>>>>>>>> y %r %r " % ( y.shape , y.mean(axis = 0))

    for itera in range(1000):
        run_iteration(train, iteration = itera, size = size * (itera+1) )

    print gen_model.summary()
    
if __name__== "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
