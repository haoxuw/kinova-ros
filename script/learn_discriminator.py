__N__ = -1
__BATCH__ = 8
__EPOCH__ = 8
__TEST_RATIO__ = 0.1

__MAX_TRAJS__ = 1e9
__TEST_SIZE__ = 1000

epoch = __EPOCH__

from process_scripts import *

import sys, datetime
from tensorflow import keras

__DISC_MODEL_NAME__ = "hw_disc"
__GENE_MODEL_NAME__ = "hw_gene"
__AUTO_MODEL_NAME__ = "hw_auto"
__DISC_FOLDER__ = "/disc/"
__GENE_FOLDER__ = "/gene/"
__AUTO_FOLDER__ = "/auto/"



def HW_Conv1D(input_shape):
    __NAME__ = __DISC_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"
    __FACTOR__ = 4
    __FILTER_SIZE__ = 7

    activation = 'relu'

    def add_repetition_unit_v1(x, filters, filter_size, j):
        x = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREFIX__+ 'cv1_%d_x_%d' %(filters, j) )(x)
        return x

    def add_repetition_unit_v2(x, filters, filter_size, j):
        ori_x = x

        x1 = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREFIX__+ 'cv1_%d_x_%d' %(filters, j) )(x)

        x2 = x
        x2 = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREFIX__+ 'cv1_21_%d_x_%d' %(filters, j) )(x2)
        x2 = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREFIX__+ 'cv1_22_%d_x_%d' %(filters, j), padding = 'same' )(x2)
        x2 = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREFIX__+ 'cv1_23_%d_x_%d' %(filters, j), padding = 'same' )(x2)

        x3 = x
        x3 = keras.layers.Conv1D(filters, filter_size, activation=activation, name = __NAME_PREFIX__+ 'cv1_3_%d_x_%d' %(filters, j) )(x3)
        x3 = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense31_%d_num_%d" %(filters, j) )(x3)
        x3 = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense32_%d_num_%d" %(filters, j) )(x3)
        x3 = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense33_%d_num_%d" %(filters, j) )(x3)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' %(j))([x1, x2, x3])

        return x

    conv_shape = np.array([128,192,256,256,256,256,256,512,512,512])
    filter_size = __FILTER_SIZE__

    stride = 1
    padding = 0
    last_layer_x = input_shape[1]

    entry = keras.layers.Input(shape=input_shape[1:], name = __NAME_PREFIX__ + "entry")

    x = entry

    for j in range(len(conv_shape)):
        filters = conv_shape[j]
        last_layer_x = (last_layer_x - filter_size + 2 * padding) / stride + 1
        #print last_layer_x
        if last_layer_x < 1:
            print "last_layer_x %d < 1" % last_layer_x
            sys.exit(last_layer_x)
        x = add_repetition_unit_v1(x, filters, filter_size, j)

    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    dim = conv_shape[-1] * input_shape[2] # last_layer_x * 
    dense_shape = []

    activation = 'sigmoid'
    activation = 'linear'
    cnt = 0
    while dim > 4:
        dim /= __FACTOR__
        x = keras.layers.Dense(dim, activation=activation, name = __NAME_PREFIX__ + "dense_" + str(cnt) )(x)
        cnt += 1

    exit = keras.layers.Dense(1, activation='sigmoid', name = __NAME_PREFIX__ + "exit" )(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()

    return model

def create_discriminator(input_shape):
    model = HW_Conv1D(input_shape)
    model.compile(optimizer='adam',
                  loss='MAE',
                  metrics=['MAE']
    )
    model.build(input_shape)
    return model


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=0.00001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("\nEarly stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print "\n\nEpoch %05d: early stopping THR" % epoch
            self.model.stop_training = True

def fit_and_save_model(model, train, test, model_output_path, save_model = True, epoch = __EPOCH__, visualize = True):
    print train[0].shape
    print train[1].shape

    print " -- training model -- "

    random_vis = 100#10
    for i in range(random_vis):
        index = np.random.randint(train[0].shape[0])
        visualize_script(train[0][index], dist = train[1][index])




    checkpoint = model_output_path + model.name + "_cp.hdf5"

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint, verbose=1, save_best_only=True, mode='max')

    log_dir = model_output_path + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=0.001, verbose=1)

    callbacks = [cp_callback, tensorboard_callback, stopping_callback]
    callbacks = [tensorboard_callback, stopping_callback]

    fit = model.fit(train[0], train[1], epochs=epoch, batch_size=__BATCH__, validation_split=(__TEST_RATIO__), callbacks=callbacks)

    #model.save(model_output_path + model.name + '.tf', save_format="tf")
    if save_model:
        model.save(model_output_path + model.name + '.h5')

    if test:
        model.evaluate(*test)

        if visualize:
            answer = model.predict(test[0])
            for index in range(15):#range(test[0].shape[0]):
                print "%d As %r" % (index, answer[index])
                print "%d Vs %r" % (index, test[1][index])

                print answer[index].shape
                visualize_script(test[0][index], dist = 'test')
                visualize_script(answer[index], dist = 'pred')
                filename = output_path + '/fit_pred_sample_%d.png' % index
                visualize_script(answer[index], dist = 'pred', filename = filename)
                filename = output_path + '/fit_test_sample_%d.png' % index
                visualize_script(test[0][index], dist = 'test', filename = filename)

    return model

def load_np_data(path, chop_time = True, max_size = -1, visualize = False):

    if chop_time:
        x_shape = [-1, 64, 7]
    else:
        x_shape = [-1, 64, 6]
    y_shape = [-1, 1]

    x_train = load_from_npy(path + "x_train.npy", x_shape, max_size)
    y_train = load_from_npy(path + "y_train.npy", y_shape, max_size)

    affine_train = load_from_npy(path + "affine_train.npy", x_shape, max_size)

    print
    print "Loaded data x.shape: %r, y.mean: %r" % ( x_train.shape, y_train.mean())
    print
    print
    print

    x_test = load_from_npy(path + "x_test.npy", x_shape, max_size/10)
    y_test = load_from_npy(path + "y_test.npy", y_shape, max_size/10)

    affine_test = load_from_npy(path + "affine_test.npy", x_shape, max_size/10)

    if visualize:
        for i in range(0, len(x_train), len(x_train)/7):
            x = x_train[i]
            y = y_train[i]
            visualize_script(x, dist = y)
        for i in range(0, len(affine_test), len(affine_test)/10):
            x = affine_test[i]
            visualize_script(x, dist = 0.001)

    if chop_time:
        x_train = x_train[:,:,1:]
        x_test = x_test[:,:,1:]
        affine_train = affine_train[:,:,1:]
        affine_test = affine_test[:,:,1:]


    test = (x_test, y_test)
    train = (x_train, y_train)

    return test, train, affine_test, affine_train

def create_and_fit_discriminator(train, test, path):

    print "shuffle pool %r" % (train[0].shape[0]*2)
    #train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(train[0].shape[0]*2).batch(__BATCH__)
    #test_ds = tf.data.Dataset.from_tensor_slices(test).batch(__BATCH__)

    model = create_discriminator(train[0].shape)
    fit_and_save_model(model, train, test, path, epoch = __EPOCH__, visualize = False)

def main():
    max_size = 100
    test, train, _, _ = load_np_data(output_path, visualize = True, max_size = max_size) #True)


if __name__== "__main__":
    main()
