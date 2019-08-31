__N__ = -1
__BATCH__ = 100
__EPOCH__ = 20
__TEST_RATIO__ = 0.1

__MAX_TRAJS__ = 5000000
__TEST_SIZE__ = 1000

from process_scripts import *

import sys, datetime
import tensorflow as tf
from tensorflow.keras.layers import *

class HW_Conv1D(tf.keras.Model):
    def __init__(self):
        super(HW_Conv1D, self).__init__()

        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(1, activation='relu')

        shape = np.array([128,192,256,512,768,1024,2048])
        filter_size = 9

        self.conv_pos = []
        for i in shape:
            self.conv_pos.append(Conv1D( i, filter_size, activation='relu'))


    def call(self, x):

        x = tf.identity(x, name = "input")

        x0 = x[:,:-1]
        x1 = x[:,1:]

        v = x1 - x0

        v0 = v[:,:-1]
        v1 = v[:,1:]

        a = v1 - v0

        v = v0
        x = x[:,:-2]

        concat = tf.concat([x,v,a], 2)
        x = concat

        for layer in self.conv_pos:
            x = layer(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


def create_model(shape):
    return HW_Conv1D()

def run_model(model, train, test, model_output_path):
    model.compile(optimizer='adam',
                  loss='MAE',
                  #metrics=['mean_absolute_error']
    )
    #print train[0].shape
    #print train[1].shape


    checkpoint_dir = os.path.dirname(model_output_path + model.name + "_cp.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_output_path, save_weights_only=True, verbose=1)

    log_dir = model_output_path + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    

    fit = model.fit(*train, epochs=__EPOCH__, batch_size=__BATCH__, validation_split=(1-__TEST_RATIO__), callbacks=[cp_callback, tensorboard_callback])

    model.save(model_output_path + model.name + '.tf', save_format="tf")

    model.evaluate(*test)

    index = 5
    answer = model.predict(test[0])
    print "Predict " + str(test[0][index])
    print "As " + str(answer[index])
    print "Vs " + str(test[1][index])

__RESOLUTION__ = 64
__FIXED_RANGE__ = [
    [0, 32, 100], #dummy
    [-0.7, 0.1, __RESOLUTION__],
    [-0.8, 0, __RESOLUTION__],
    [0.1, 0.7, __RESOLUTION__],
]

def report_stat(scripts):
    print scripts.shape
    for index in range(0,4):
        rang = [np.amin(scripts[:,:,index]), np.amax(scripts[:,:,index])]
        if rang[0] < __FIXED_RANGE__[index][0] or rang[1] > __FIXED_RANGE__[index][1]:
            print "got " + str(rang) + " vs __FIXED_RANGE__ " + str(__FIXED_RANGE__)
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

    model = create_model(scripts[0].shape)
    run_model(model, train, test, path)


def learn_trajs(path):
    max = __MAX_TRAJS__
    x_train = load_from_npy(path + "x_train.npy")[:max]
    y_train = load_from_npy(path + "y_train.npy")[:max]
    x_test = load_from_npy(path + "x_test.npy")[:max/10]
    y_test = load_from_npy(path + "y_test.npy")[:max/10]
    print "Training on x.shape: %r, y.mean: %r" % ( x_train.shape, y_train.mean())

    test = (x_test, y_test)
    train = (x_train, y_train)
    train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(train[0].shape[0]*2).batch(__BATCH__)
    test_ds = tf.data.Dataset.from_tensor_slices(test).batch(__BATCH__)

    model = create_model(x_train.shape)
    run_model(model, train, test, path)

#also defined in process.py
output_path = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/mini_results/"
output_path = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/fake_results/"
def main():
    learn_trajs(output_path)


if __name__== "__main__":
    main()
