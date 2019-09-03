__N__ = -1
__BATCH__ = 1#32
__EPOCH__ = 3
__TEST_RATIO__ = 0.1

__MAX_TRAJS__ = 1e9
__TEST_SIZE__ = 1000

from process_scripts import *

import sys, datetime
import tensorflow as tf

from tensorflow import keras

def HW_Conv1D(input_shape):
    __NAME__ = 'hw_disc'
    __NAME_PREVIX__ = __NAME__ + "_"
    __FACTOR__ = 4
    __FILTER_SIZE__ = 9

    conv_shape = np.array([128,192,256,512,768,1024,2048])
    filter_size = __FILTER_SIZE__

    stride = 1
    padding = 0
    last_layer_x = input_shape[1]

    entry = keras.Input(shape=input_shape[1:], name = __NAME_PREVIX__ + "entry")

    x = entry

    for i in conv_shape:
        last_layer_x = (last_layer_x - filter_size + 2 * padding) / stride + 1
        #print last_layer_x
        if last_layer_x < 1:
            sys.exit(last_layer_x)
        x = keras.layers.Conv1D(i, filter_size, activation='relu', name = __NAME_PREVIX__+ 'conv1d_%d_x_%d' %(last_layer_x, i) )(x)

    x = keras.layers.Flatten(name = __NAME_PREVIX__ + "flatten")(x)

    dim = conv_shape[-1] * input_shape[2] # last_layer_x * 
    dense_shape = []

    cnt = 0
    while dim > 4:
        dim /= __FACTOR__
        x = keras.layers.Dense(dim, activation='relu', name = __NAME_PREVIX__ + "dense_" + str(cnt) )(x)
        cnt += 1
    exit = keras.layers.Dense(1, activation='sigmoid', name = __NAME_PREVIX__ + "exit" )(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()

    return model

def create_discriminator(input_shape):
    model = HW_Conv1D(input_shape)
    model.compile(optimizer='adam',
                  loss='MSE',
                  metrics=['MAE']
    )
    model.build(input_shape)
    return model

def fit_model(model, train, test, model_output_path, save_model = True):
    #print train[0].shape
    #print train[1].shape

    checkpoint_dir = os.path.dirname(model_output_path + model.name + "_cp.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_output_path, save_weights_only=True, verbose=1)

    log_dir = model_output_path + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    

    fit = model.fit(*train, epochs=__EPOCH__, batch_size=__BATCH__, validation_split=(__TEST_RATIO__), callbacks=[cp_callback, tensorboard_callback])

    #model.save(model_output_path + model.name + '.tf', save_format="tf")
    if save_model:
        model.save(model_output_path + model.name + '.h5')

    if test:
        model.evaluate(*test)

        index = 5
        answer = model.predict(test[0])
        print "Predict " + str(test[0][index])
        print "As " + str(answer[index])
        print "Vs " + str(test[1][index])

    return model



def report_stat(scripts):
    print scripts.shape
    for index in range(0,4):
        rang = [np.amin(scripts[:,:,index]), np.amax(scripts[:,:,index])]
        if rang[0] < fixed_range[index][0] or rang[1] > fixed_range[index][1]:
            print "got " + str(rang) + " vs fixed_range " + str(fixed_range)
            sys.exit(-6)
        print "Range: " + str(rang)
    
def translate_into_3D(scripts):
    report_stat(scripts)
    for script in scripts:
        visualize_script(script)
    return scripts


def learn_trajs_from_task_files(path):

    names, dists, scripts = load_task_file_under_path(path, __N__);

    features = scripts[:,:,1:]
    #features = translate_into_3D(scripts)

    labels = dists
    assert(features.shape[0] == labels.shape[0])
    size = features.shape[0]

    test_size = int(__TEST_SIZE__)
    test = (features[:test_size], labels[:test_size])
    train = (features[test_size:], labels[test_size:])
    train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(size).batch(__BATCH__)
    test_ds = tf.data.Dataset.from_tensor_slices(test).batch(__BATCH__)

    model = create_discriminator(scripts[0].shape)
    fit_model(model, train, test, path)

    return

def load_np_data(path, chop_time = True):
    max_traj = int(__MAX_TRAJS__)
    x_train = load_from_npy(path + "x_train.npy")[:max_traj]
    if chop_time:
        x_train = x_train[:,:,1:]

    y_train = load_from_npy(path + "y_train.npy")[:max_traj]

    print
    print "Loaded data x.shape: %r, y.mean: %r" % ( x_train.shape, y_train.mean())
    print
    print
    print

    x_test = load_from_npy(path + "x_test.npy")[:max_traj/10]
    if chop_time:
        x_test = x_test[:,:,1:]
    y_test = load_from_npy(path + "y_test.npy")[:max_traj/10]

    test = (x_test, y_test)
    train = (x_train, y_train)

    return test,train

def learn_trajs(path, test, train):

    print "shuffle pool %r" % train[0].shape[0]*2
    train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(train[0].shape[0]*2).batch(__BATCH__)
    test_ds = tf.data.Dataset.from_tensor_slices(test).batch(__BATCH__)

    model = create_discriminator(train[0].shape)
    fit_model(model, train, test, path)

def main():
    test,train = load_np_data(output_path)
    learn_trajs(output_path, test, train)


if __name__== "__main__":
    main()
