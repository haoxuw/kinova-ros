from learn_discriminator import *
from process_scripts import *

import sys, datetime
import tensorflow as tf
from tensorflow.keras.layers import *

def HW_Gen_CONV(discriminator):
    __NAME__ = 'hw_gen'
    __NAME_PREVIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 9

    __DECONV_SHAPE__ = [32, 64, 128, 256, 512, 512, 1024, 2048]

    seed = keras.Input(shape=(3), name = __NAME_PREVIX__ + "seed")

    [entry, exit, discriminator_model] = discriminator
    disc_input_shape = entry.input_shape[0][1:]
    print disc_input_shape

    dense_shape = [6]
    #dense_shape.append(disc_input_shape[0] * disc_input_shape[1])
    x = seed
    for cnt,filters in enumerate(dense_shape):
        x = keras.layers.Dense(filters, activation='relu', name = __NAME_PREVIX__ + "dense_" + str(cnt) )(x)

    x = tf.keras.layers.Reshape([1, 1, dense_shape[-1]])(x)

    deconv_shape = __DECONV_SHAPE__

    kernel_size = (__FILTER_SIZE__ , 1)

    cnt = 1
    for cnt, filters in enumerate(deconv_shape):
        layer = keras.layers.Conv2DTranspose ( filters, kernel_size, activation='relu',
                                               name = __NAME_PREVIX__ + "transpseconv_" + str(cnt) )
        x = layer(x)

    print          x
    x = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)), name = __NAME_PREVIX__ + "cropping")(x)

    cnt = 1
    last_layer_channel = deconv_shape[-1] / 2
    while last_layer_channel > 6:
        layer = keras.layers.Conv2D (last_layer_channel, [1,1], activation='relu',
                                     name = __NAME_PREVIX__ + "1X1_conv_" + str(cnt) )
        x = layer(x)
        cnt += 1

        last_layer_channel /= 2

    x = keras.layers.Conv2D (6, [1,1],
                             activation = 'sigmoid', # 'relu',
                             name = __NAME_PREVIX__ + "1X1_conv_last" )(x)
    x = tf.keras.layers.Reshape([64, 6], name = __NAME_PREVIX__ + "script")(x)

    #x = x / 6.0

    x = x * np.array([fixed_range[1][1] - fixed_range[1][0],
                      fixed_range[2][1] - fixed_range[2][0],
                      fixed_range[3][1] - fixed_range[3][0],
                      fixed_range[4][1] - fixed_range[4][0],
                      fixed_range[5][1] - fixed_range[5][0],
                      fixed_range[6][1] - fixed_range[6][0],
    ])

    x = x + np.array([fixed_range[1][0],
                      fixed_range[2][0],
                      fixed_range[3][0],
                      fixed_range[4][0],
                      fixed_range[5][0],
                      fixed_range[6][0],
    ])

    script_tensor = x

    pos = tf.keras.layers.Reshape([64, 3])(script_tensor[:,:,0:3])

    avg_pos = tf.keras.layers.AveragePooling1D (pool_size = 64)(pos)

    discriminator_loss = discriminator_model(script_tensor)

    avg_pos_loss = keras.layers.Subtract(name = __NAME_PREVIX__ + "avg_pos_loss_layer") ([seed, avg_pos])

    avg_pos_loss = tf.keras.backend.mean(avg_pos_loss, axis = 2) * 375

    #loss = (discriminator_loss + avg_pos_loss)
    loss = discriminator_loss
    #loss = avg_pos_loss

    #loss = tf.keras.layers.Multiply(name = __NAME_PREVIX__ + "loss_layer_output") ([discriminator_loss, avg_pos_loss])
    loss = tf.keras.layers.Multiply(name = __NAME_PREVIX__ + "loss_layer_output") ([loss, loss])

    '''
    discriminator_loss = tf.constant([1] * 10, dtype= float)
    discriminator_loss = tf.keras.layers.Reshape([1])(discriminator_loss)
    print discriminator_loss

    l = tf.keras.layers.Multiply(name = __NAME_PREVIX__ + "loss_layer_output")
    loss = l ([discriminator_loss, avg_pos_loss])
    print l
    print l.name
    print loss
    print 11111
    print 11111
    print 11111
    print 11111
    print 11111
    '''
    
    model = keras.Model(inputs=seed, outputs=[loss,script_tensor] , name= __NAME__)

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

def main():
    name = "hw"

    #test, train = load_np_data(output_path, False)

    discr = load_discriminator(output_path, name)

    model = create_generator( discr )

    size = 1000#00
    data = [np.random.rand(size, 3) , np.zeros(size)]
    data[0][:,0] -= 0.8
    data[0][:,1] -= 0.8
    data[0][:,2] *= 0.5
    data[0][:,2] += 0.2

    print data[0]

    model = fit_model(model, data, None, output_path + "/generator/")


    size = 100
    data = np.random.rand(size, 3)
    for index in range(data.shape[0]):
        seed = np.array(data[index,:]).reshape(1,-1)
        [losses, scripts] = model.predict(seed)
        script = scripts[0]
        script = add_time_axis(script)
        print script
        visualize_script(script)

    return


if __name__== "__main__":
    main()
