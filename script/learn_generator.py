from learn_discriminator import *
from process_scripts import *

import sys, datetime

from tensorflow import keras

from arg_parser import *











def HW_Disc_Dense(input_shape):
    __NAME__ = args.__DISC_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"
 
    activation = 'relu'
    activation = 'tanh'

    dense_shape = np.array([12,32,128,128,256,128,128,32,16,4])
    #dense_shape = np.array([12,256,1024,512]) v1

    __FILTER_SIZE__ = 3
    

    def add_disc_repetition_v1(x, ori_x, filters, j):
        x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten_%d" % (j))(x)
        x = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense_" + str(j) )(x)

        return x

    def add_disc_repetition_v2(x, ori_x, filters, j):
        filter_size = __FILTER_SIZE__
        factor = 2

        x1 = keras.layers.Conv1D(filters, filter_size, activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'cv1_1_%d_x_%d' %(filters, j) )(x)

        input_x_size = x.shape[1]

        ori_x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten_%d" % (j))(ori_x)
        x2 = keras.layers.Dense(filters * input_x_size * factor, activation=activation, name = __NAME_PREFIX__ + "dense_21_" + str(j) )(ori_x)
        x2 = keras.layers.Dense(filters * input_x_size, activation=activation, name = __NAME_PREFIX__ + "dense_22_" + str(j) )(x2)
        x2 = keras.layers.Reshape([input_x_size, filters])(x2)

        x3 = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense_3_" + str(j) )(x)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' % (j))([x1, x2, x3])

        if (j % 2 == 0):
            x = keras.layers.MaxPooling1D(name = __NAME_PREFIX__+ 'maxpool_num_%d' % (j), pool_size = 2)(x)

        return x


    entry = keras.layers.Input(shape=input_shape[1:], name = __NAME_PREFIX__ + "entry")

    x = entry

    v1 = keras.layers.Cropping1D(cropping=([1, 0]), name = __NAME_PREFIX__ + "cropping_v1" )(x)
    v2 = keras.layers.Cropping1D(cropping=([0, 1]), name = __NAME_PREFIX__ + "cropping_v2" )(x)
    v = keras.layers.Subtract(name = __NAME_PREFIX__+ 'sub_v')([v2,v1])

    x = keras.layers.Concatenate(name = __NAME_PREFIX__+ 'concat_v', axis = 2)([v,v2])

    
    ori_x = x

    for j in range(len(dense_shape)):
        filters = dense_shape[j]
        x = add_disc_repetition_v2(x, ori_x, filters, j)

    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    x = keras.layers.Dense(1, activation='linear', name = __NAME_PREFIX__ + "shift" )(x)

    exit = keras.layers.Dense(1, activation='sigmoid', name = __NAME_PREFIX__ + "exit" )(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()

    return model



def HW_Disc_Conv1D(input_shape):
    __NAME__ = args.__DISC_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"
    __FACTOR__ = 4
    __FILTER_SIZE__ = 3

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
        x = x3

        return x

    conv_shape = np.array([12,12,12,12,12,12,64,64,64,64,64,128,128,128,128,128,128,128,128,128,128,128,256,])
    filter_size = args.__FILTER_SIZE__

    stride = 1
    padding = 0
    last_layer_x = input_shape[1]

    entry = keras.layers.Input(shape=input_shape[1:], name = __NAME_PREFIX__ + "entry")

    x = entry

    v1 = keras.layers.Cropping1D(cropping=([1, 0]), name = __NAME_PREFIX__ + "cropping_v1" )(x)
    v2 = keras.layers.Cropping1D(cropping=([0, 1]), name = __NAME_PREFIX__ + "cropping_v2" )(x)
    v = keras.layers.Subtract(name = __NAME_PREFIX__+ 'sub_v')([v2,v1])
    print x
    print v
    print v2
    x = keras.layers.Concatenate(name = __NAME_PREFIX__+ 'concat_v', axis = 2)([v,v2])
    print x

    for j in range(len(conv_shape)):
        filters = conv_shape[j]
        last_layer_x = (last_layer_x - filter_size + 2 * padding) / stride + 1
        #print last_layer_x
        if last_layer_x < 1:
            print "last_layer_x %d < 1" % last_layer_x
            sys.exit(last_layer_x)
        x = add_repetition_unit_v2(x, filters, filter_size, j)

    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    dim = conv_shape[-1] * input_shape[2] # last_layer_x * 
    dense_shape = []

    activation = 'sigmoid'
    activation = 'linear'
    cnt = 0
    while dim > 4:
        dim /= args.__FACTOR__
        x = keras.layers.Dense(dim, activation=activation, name = __NAME_PREFIX__ + "dense_" + str(cnt) )(x)
        cnt += 1

    exit = keras.layers.Dense(1, activation='sigmoid', name = __NAME_PREFIX__ + "exit" )(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()

    return model

def create_discriminator(input_shape):
    #model = HW_Disc_Conv1D(input_shape)
    model = HW_Disc_Dense(input_shape)
    model.compile(optimizer='adam',
                  loss='MSE',
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

def fit_and_save_model(model, train, test, model_output_path, save_model = True, epochs = 1, visualize = True):
    print " -- training model -- "

    checkpoint = model_output_path + model.name + "_cp.hdf5"

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint, verbose=1, save_best_only=True, mode='max') #monitor='val_acc'

    log_dir = model_output_path + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=0.001, verbose=1)

    callbacks = [cp_callback, tensorboard_callback, stopping_callback]
    #callbacks = [tensorboard_callback, stopping_callback]

    fit = model.fit(train[0], train[1], epochs = epochs, batch_size= args.batch, validation_split=( args.__TEST_RATIO__), callbacks=callbacks)

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
                filename = model_output_path + '/fit_pred_sample_%d.png' % index
                visualize_script(answer[index], dist = 'pred', filename = filename)
                filename = model_output_path + '/fit_test_sample_%d.png' % index
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

    print "x_test.shape"
    print x_test.shape
    print "[:,0,:] point range:"
    print "min:"
    print x_test[:,0,:].min(axis = 0)
    print "max:"
    print x_test[:,0,:].max(axis = 0)

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
    fit_and_save_model(model, train, test, path, epochs = args.epochs, visualize = False)


def HW_decoder(input_shape):
    __NAME__ = args.__DECO_MODEL_NAME__
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
    activation = 'tanh'

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

    x = keras.layers.Reshape((64,6), name = args.__DECO_MODEL_NAME__ + "_exit")(x)

    exit = x

    model = keras.Model(inputs=seed, outputs=exit , name= __NAME__)

    model.compile(optimizer = 'adam',
                  loss = 'MSE',
                  )

    print model.summary()

    return model


def HW_Gene_Minimal_GAN(decoder_pack, discriminator_pack):
    __NAME__ = args.__GENE_MODEL_NAME__
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

    gene_model = keras.Model(inputs=seed, outputs=[disc_loss, script] , name= args.__GENE_MODEL_NAME__)

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
        if args.__DECO_MODEL_NAME__ == layer.name:
            print "-------- configure %s : %r  ----------" % (layer.name, deco_trainable)
            layer.trainable = deco_trainable
        elif args.__DISC_MODEL_NAME__ == layer.name:
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

    config_generator (gene_model, deco_trainable = False, disc_trainable = False)
    mini_size = 3
    affine_data = [ np.zeros(mini_size * 6).reshape(mini_size, 1, 6), np.zeros(mini_size).reshape(-1, 1) ]
    gene_model = fit_and_save_model(gene_model, train = affine_data, test = None, model_output_path = args.output_dir + args.__GENE_FOLDER__, epochs = args.epochs, visualize = False)

    return gene_model


def load_model_pack(path, model_name, trainable = False, entry = "entry", exit = "exit"):
    model = keras.models.load_model(path + "/" + model_name + ".h5")
    model.load_weights( path + "/" + model_name + "_cp.hdf5" )
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

    fit_and_save_model(deco_model, affine_data_train, affine_data_test, path, epochs = args.epochs, visualize = False)

    return deco_model

def fit_generator_decoder(gene_model, demo_data, path, size = 10000):
    print "\n\n-------- re-train decoder with sampled demo+random seed ----------\n\n"

    seeds = gen_seeds(size)
    losses = np.zeros(size).reshape(-1,1)

    random_data = [seeds , losses]
    affine_data = [ demo_data[:,0:1,:], np.zeros(demo_data.shape[0]).reshape(-1,1) ]

    sampled_data = merge_data(random_data, affine_data, size, 0.5)

    config_generator (gene_model, deco_trainable = True, disc_trainable = False)
    gene_model = fit_and_save_model(gene_model, train = sampled_data, test = None, model_output_path = path, epochs = args.epochs, visualize = False)

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
    gene_model = fit_and_save_model(gene_model, train = sampled_data, test = None, model_output_path = path, epochs = args.epochs, visualize = False)

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

def save_sample_figures(scripts, save_path, prefix = "sample_fig", size = 50):
    max_size = 100;
    if save_path is not None:
        path = args.output_dir + "/" + save_path
        os.system('mkdir -p ' + path)

        total_size = scripts.shape[0]
        for sample_index in range(0, total_size, (total_size / size + 1)):
            script = scripts[sample_index]
            sample_script = np.copy(script).reshape([64,6])

            description = ("#%02d" % sample_index) + "_" + prefix
            filename = path + "/" + description
            visualize_script(sample_script, filename = filename, dist = description, dequant = True, write_traj = True)

            max_size -= 1
            if max_size == 0:
                break

def load_generator_cp_pack():
    return load_model_pack(args.output_dir + args.__GENE_FOLDER__, args.__GENE_MODEL_NAME__, trainable = None)

def run_iteration(gene_model, affine_train, affine_test, iteration = 0, size = 10000, path = args.__GENE_FOLDER__, save_samples_path = args.__FIG_FOLDER__):
    print "\n\n-------- Running Iteration %d  ----------\n\n" % iteration
    #need to load model every iteration to workaround what seems to be a tensorflow mem leak

    print "\n\n-------- acquired generator  ----------\n\n"
    if iteration == 0:
        print gene_model.summary()

        prefix = "Demonstration"
        save_sample_figures(affine_test, save_samples_path, prefix)

    gene_model, gen_predicted_data = fit_generator_decoder(gene_model = gene_model, demo_data = affine_train, path = path, size = size)

    prefix = "Predicted_In_Iteration_" + str(iteration)

    test_seeds = affine_test[:,:1,:]
    gen_predicted_test_data = predict_scripts(gene_model, test_seeds)
    save_sample_figures(gen_predicted_test_data[0], save_samples_path, prefix)

    gene_model, gen_predicted_data = fit_generator_discriminator(gene_model = gene_model, demo_data = affine_train, gen_predicted_data = gen_predicted_data, path = path, size = size)

    #prefix = "posdisc_iter_" + str(iteration)

    return gene_model

def main():
    size = args.max_size

    print "************ Running Experiement Iteration %d, re_train == %r ************\n\n\n" % (args.itera, args.re_train)
    
    print "\n\n-------- loading train/test dataset  ----------\n\n"
    test, train, affine_test, affine_train = load_np_data(args.output_dir, max_size = size, visualize = False)
    print "\ttrain size %d, affine_train size %d" % (train[0].shape[0], affine_train[0].shape[0])

    if args.re_train:
        print "\n\n-------- load_generator  ----------\n\n"
        gene_model = load_model_pack(args.output_dir + args.__GENE_FOLDER__, args.__GENE_MODEL_NAME__, trainable = None)[2]

        print "\n\n-------- continue training generator  ----------\n\n"
        gene_model = run_iteration(gene_model = gene_model, affine_train = affine_train, affine_test = affine_test, iteration = args.itera, path = args.output_dir + args.__GENE_FOLDER__, size = size )

    else:
        print "\n\n-------- create_and_fit_discriminator  ----------\n\n"
        create_and_fit_discriminator(train = train, test = test, path = args.output_dir + args.__DISC_FOLDER__)

        print "\n\n-------- create_and_fit_decoder  ----------\n\n"
        create_and_fit_decoder (affine_train = affine_train, affine_test = affine_test, path = args.output_dir + args.__DECO_FOLDER__)

        print "\n\n-------- load_decoder  ----------\n\n"
        decoder_pack = load_model_pack (path = args.output_dir + args.__DECO_FOLDER__, model_name = args.__DECO_MODEL_NAME__, entry = args.__DECO_MODEL_NAME__ + "_entry", exit = args.__DECO_MODEL_NAME__ + "_exit") 

        print "\n\n-------- load_discriminator  ----------\n\n"
        discr_pack = load_model_pack(path = args.output_dir + args.__DISC_FOLDER__, model_name = args.__DISC_MODEL_NAME__)

        print "\n\n-------- create_generator  ----------\n\n"
        gene_model = create_generator( decoder_pack = decoder_pack, discriminator_pack = discr_pack )

    return gene_model
    
if __name__== "__main__":
    args = argumentParser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
