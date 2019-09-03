from learn_discriminator import *
from process_scripts import *

import sys, datetime
import tensorflow as tf
from tensorflow.keras.layers import *

def normalize_into(x, x_range):
    x = x * np.array([x_range[1][1] - x_range[1][0],
                      x_range[2][1] - x_range[2][0],
                      x_range[3][1] - x_range[3][0],
                      x_range[4][1] - x_range[4][0],
                      x_range[5][1] - x_range[5][0],
                      x_range[6][1] - x_range[6][0],
    ])

    x = x + np.array([x_range[1][0],
                      x_range[2][0],
                      x_range[3][0],
                      x_range[4][0],
                      x_range[5][0],
                      x_range[6][0],
    ])

    return x
    
def HW_Gen_CONV(discriminator):
    __NAME__ = 'hw_gen'
    __NAME_PREVIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 9

    __DECONV_SHAPE__ = [32, 64, 128, 256, 512, 512, 1024, 2048]
    __DENSE_SHAPE__ = [6, 32]

    seed = keras.Input(shape=(3), name = __NAME_PREVIX__ + "seed")
    x = seed

    [entry, exit, discriminator_model] = discriminator
    disc_input_shape = entry.input_shape[0][1:]
    print disc_input_shape

    dense_shape = __DENSE_SHAPE__
    activation = 'relu'
    activation = 'linear'
    activation = 'sigmoid'

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

    x = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)), name = __NAME_PREVIX__ + "cropping")(x)

    cnt = 1
    last_layer_channel = deconv_shape[-1] / 2
    while last_layer_channel > 6:
        layer = keras.layers.Conv2D (last_layer_channel, [1,1], activation=activation,
                                     name = __NAME_PREVIX__ + "1X1_conv_" + str(cnt) )
        x = layer(x)
        cnt += 1

        last_layer_channel /= 2

    x = keras.layers.Conv2D (6, [1,1],
                             activation = activation,
                             name = __NAME_PREVIX__ + "1X1_conv_last" )(x)


    x = tf.keras.layers.Reshape(disc_input_shape, name = __NAME_PREVIX__ + "script")(x)

    x = normalize_into(x, fixed_range)

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

def load_discriminator(path, model_name):
    model = tf.keras.models.load_model(path + "/" + model_name + ".h5")
    model.trainable = False

    model.compile(optimizer='adam',
                  loss='MAE',
    )

    print "Loaded model %s" % model.name

    discri = [None, None, model]
    for layer in model.layers:
        layer.trainable = False
        if "entry" in layer.name:
            discri[0] = layer
        elif "exit" in layer.name:
            discri[1] = layer

    print model.summary()

    return discri

def test_discriminator(disc_name):
    discr = load_discriminator(output_path, disc_name)
    discr_model = discr[2]
    test, train = load_np_data(output_path, )

    test_x = test[0][:1000]
    test_y = test[1][:1000]
    answer = discr_model.predict(test_x)
    print "!!!!!!!!!"
    print test_x.shape

    for index in range(0,test_x.shape[0], 10):
        print "ground  Y : %r" % test_y[index]
        #print "As " + str(answer[index])
        print "predict Y : %r" % answer[index]
        script = test_x[index]
        print script.shape
        script = add_time_axis(script)
        visualize_script(script)


    return

def main():
    disc_name = "hw_disc"

    #test_discriminator(disc_name)

    discr = load_discriminator(output_path, disc_name)
    discr_model = discr[2]

    model = create_generator( discr )

    size = 100#000#00
    data_x = normalize_into(np.random.rand(size, 6), fixed_range)[:,:3]
    data = [data_x , np.zeros(size)]

    print data[0]

    model = fit_model(model, data, None, output_path + "/generator/")


    size = 10000
    size = 10
    data = normalize_into(np.random.rand(size, 6), fixed_range)[:,:3]
    for index in range(data.shape[0]):
        seed = np.array(data[index,:]).reshape(1,-1)
        [losses, scripts] = model.predict(seed)
        #[losses, avg_pos] = model.predict(seed)
        script = scripts[0]
        pred_script = np.copy(script)
        script = add_time_axis(script)
        print
        print
        print "For seed"
        print seed
        print "script:"
        print script.mean(axis = 0)
        print script
        visualize_script(script)
        [answer] = discr_model.predict(pred_script.reshape(1,64,6))
        print 55555
        print answer

    return


if __name__== "__main__":
    main()
