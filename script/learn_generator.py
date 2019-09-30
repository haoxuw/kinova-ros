import gc

from process_scripts import *

import sys, datetime

from tensorflow import keras

from arg_parser import *

from keras import regularizers


#adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.999, beta_2=0.9999, amsgrad=False)
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999, amsgrad=False)






def HW_Disc_Dense(input_shape):
    __NAME__ = args.__DISC_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"
 
    activation = 'relu'
    activation = 'tanh'

    dense_shape = np.array([12,32,128,128,256,1287,128,64,32,8])
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

        x3 = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense_3_" + str(j) )(x)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' % (j))([x1, x3])

        if (j % 2 == 0):
            x = keras.layers.MaxPooling1D(name = __NAME_PREFIX__+ 'maxpool_num_%d' % (j), pool_size = 2)(x)

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)

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

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)

        return x

    def add_disc_repetition_v3(x, ori_x, filters, j):
        filter_size = __FILTER_SIZE__
        factor = 2

        activation= keras.layers.LeakyReLU(alpha=0.5,
                                           name = __NAME_PREFIX__+ 'act_1_%d_x_%d' %(filters, j) )

        x = keras.layers.Conv1D(filters,
                                filter_size, activation='linear',
                                padding = 'same',
                                kernel_regularizer=regularizers.l2(0.001),
                                #activity_regularizer=regularizers.l1(0.0001),
                                name = __NAME_PREFIX__+ 'cv1_1_%d_x_%d' %(filters, j) )(x)
        x = activation(x)
        
        if (j % 2 == 0):
            x = keras.layers.MaxPooling1D(name = __NAME_PREFIX__+ 'maxpool_num_%d' % (j), pool_size = 2)(x)

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)

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
        x = add_disc_repetition_v3(x, ori_x, filters, j)

    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    activation= keras.layers.LeakyReLU(alpha=0.2,
                                       name = __NAME_PREFIX__+ 'act' )

    x = keras.layers.Dense(1, activation='linear', name = __NAME_PREFIX__ + "shift" )(x)

    x = activation(x)

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
    x = keras.layers.Concatenate(name = __NAME_PREFIX__+ 'concat_v', axis = 2)([v,v2])

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
                  metrics=['MSE']
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

def fit_and_save_model(model, train, test, model_output_path, save_model = True, epochs = 1, visualize_train = False, visualize_pred = False):
    print " -- training model %r -- " % model.name

    checkpoint = model_output_path + model.name + "_cp.hdf5"

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint, verbose=1, save_best_only=True, mode='auto') #monitor='val_acc'

    log_dir = model_output_path + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=0.001, verbose=1)

    callbacks = [cp_callback, tensorboard_callback, stopping_callback]
    #callbacks = [tensorboard_callback, stopping_callback]

    batch = args.batch

    X, Y = train

    if visualize_train:
        for i in range(0, len(X), len(X)/37 + 1):
            x = X[i]
            y = Y[i]
            visualize_script(x, dist = y)

    fit = model.fit(X, Y, epochs = epochs, batch_size= batch, shuffle=True, validation_split=( args.__TEST_RATIO__), callbacks=callbacks)

    if save_model:
        model.save(model_output_path + model.name + '.h5')

    #model.save(model_output_path + model.name + '.tf', save_format="tf")
    if test:
        model.evaluate(test[0], test[1])

    if visualize_pred:
        answer = model.predict(X)

        print answer.shape
        if X.shape[1:] == [64,6]:
            #script = answer[1] # [loss, script]
            print "visualizing discri prediction"
            for cnt in range(35):#range(test[0].shape[0]):
                index = np.random.randint(X.shape[0])
                visualize_script(X[index], dist = "pred: %r" % answer[index][0])
        else:
            print "visualizing deco prediction"
            script = answer
            for cnt in range(20):#range(test[0].shape[0]):
                index = np.random.randint(X.shape[0])
                filename = model_output_path + '/decoder_pred_sample_%d.png' % cnt
                visualize_script(script[index], dist = "Decoder Prediction Example %d" % cnt, filename = filename)

                filename = model_output_path + '/decoder_ground_truth_%d.png' % cnt
                visualize_script(Y[index], dist = "Decoder Ground Truth %d" % cnt, filename = filename)

        
        return model



        answer = model.predict(X)
        for cnt in range(15):#range(test[0].shape[0]):
            index = np.random.randint(X.shape[0])
            print "%d As %r" % (index, script[index])
            print "%d Vs %r" % (index, Y[index])

            #visualize_script(X[index], dist = 'train')
            visualize_script(script[index], dist = 'pred')

            #filename = model_output_path + '/fit_train_sample_%d.png' % index
            #visualize_script(X[index], dist = 'train', filename = filename)
            filename = model_output_path + '/fit_pred_sample_%d.png' % index
            visualize_script(script[index], dist = 'pred', filename = filename)

    return model

def load_np_data(path, chop_time = True, max_size = -1, visualize = False, max_test = True, shuffle = False):

    if chop_time:
        x_shape = [-1, 64, 7]
    else:
        x_shape = [-1, 64, 6]
    y_shape = [-1, 1]

    x_train = load_from_npy(path + "x_train.npy", x_shape)
    y_train = load_from_npy(path + "y_train.npy", y_shape)
    affine_train = load_from_npy(path + "affine_train.npy", x_shape)

    x_test = load_from_npy(path + "x_test.npy", x_shape)
    y_test = load_from_npy(path + "y_test.npy", y_shape)
    affine_test = load_from_npy(path + "affine_test.npy", x_shape)

    if shuffle:
        sample_indices = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
        x_train = x_train[sample_indices]
        y_train = y_train[sample_indices]

        sample_indices = np.random.choice(affine_train.shape[0], affine_train.shape[0], replace=False)
        affine_train = affine_train[sample_indices]

        sample_indices = np.random.choice(x_test.shape[0], x_test.shape[0], replace=False)
        x_test = x_test[sample_indices]
        y_test = y_test[sample_indices]

        sample_indices = np.random.choice(affine_test.shape[0], affine_test.shape[0], replace=False)
        affine_test = affine_test[sample_indices]

    if max_size > 0:
        x_train = x_train[:max_size]
        y_train = y_train[:max_size]
        affine_train = affine_train[:max_size]
        if not max_test:
            max_size = max_size/10
            x_test = x_test[:max_size]
            y_test = y_test[:max_size]
            affine_test = affine_test[:max_size]
        else:
            max_size = affine_test.shape[0]
            x_test = x_test[:max_size]
            y_test = y_test[:max_size]

            step = max_size / 41
            x_test = x_test[0::step]
            y_test = y_test[0::step]
            affine_test = affine_test[0::step]
    gc.collect()

    print
    print

    print
    print "Loaded data x.shape: %r, y.mean: %r" % ( x_train.shape, y_train.mean())
    print
    print "x_train.shape"
    print x_train.shape
    print "[:,0,:] point range:"
    print "min:"
    print x_train[:,0,:].min(axis = 0)
    print "max:"
    print x_train[:,0,:].max(axis = 0)

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
    fit_and_save_model(model, train, test, path, epochs = args.epochs, visualize_train = False)
    return model


def HW_decoder(input_shape):
    __NAME__ = args.__DECO_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 7

    __SEED_TRANS_NUM__ = 5

    __SHAPE__ = [32, 64, 128, 256, 256, 256, 256, 128, 128, 64]


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
        s51 = keras.layers.Dense(c_size * 2 , activation=activation, name = __NAME_PREFIX__ + "dense51_%d_num_%d" %(filters, j) )(s)
        s52 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREFIX__ + "dense53_%d_num_%d" %(filters, j) )(s51)
        s5 = keras.layers.Dense(x_size * c_size, activation=activation, name = __NAME_PREFIX__ + "dense5_%d_num_%d" %(filters, j) )(s52)
        s5 = keras.layers.Reshape([x_size, y_size, c_size])(s5)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' %(j))([x1, x2, x3, s5])

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)
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

    x = keras.layers.Conv2DTranspose(6, (last_layer_x, 1), activation='sigmoid', name = __NAME_PREFIX__+ 'deconv_first' )(x)

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
        if args.__DECO_MODEL_NAME__ in layer.name:
            print "-------- configure %s : %r  ----------" % (layer.name, deco_trainable)
            layer.trainable = deco_trainable
        elif args.__DISC_MODEL_NAME__ in layer.name:
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


def extract_model_from_generator(model_name, new_model_name):
    entry, exit, gene_model = load_model_pack (path = args.output_dir + args.__GENE_FOLDER__, model_name = args.__GENE_MODEL_NAME__, trainable = True)

    submodel = None
    for layer in gene_model.layers:
        if model_name == layer.name:
            submodel = layer

    if submodel is None:
        return None

    entry = keras.layers.Input(shape=(64,6), name = new_model_name + "_entry")

    x = entry

    exit = submodel(x)

    sub_model = keras.Model(inputs=entry, outputs=exit , name= new_model_name)

    sub_model.compile(optimizer='adam',
                  loss='MSE',
    )

    print "\n\n-------- extracted (%r) as:  ----------\n\n" % model_name
    print sub_model.summary()
    
    return [entry, exit, sub_model]



def load_model_pack(path, model_name, trainable = None, entry = "entry", exit = "exit"):
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

    seeds [:,0] *= 0.8
    seeds [:,1] *= 0.7
    seeds [:,2] *= 0.17
    seeds [:,3] *= 0.33
    seeds [:,4] *= 0
    seeds [:,5] *= 0.63

    print "min:"
    print seeds.min(axis = 0)
    print "max:"
    print seeds.max(axis = 0)

    return seeds.reshape([-1,1,6])

def create_and_fit_decoder(affine_train, affine_test, path):
    deco_model = HW_decoder([None,1,affine_train.shape[-1]])

    deco_model.compile(optimizer = 'adam',
                       loss = 'MSE',
    )

    affine_data_train = [affine_train[:,0:1,:], affine_train]
    affine_data_test = [affine_test[:,0:1,:], affine_test]

    fit_and_save_model(deco_model, affine_data_train, affine_data_test, path, epochs = args.epochs, visualize_train = False, visualize_pred = False)

    return deco_model
        
def fit_generator_decoder(gene_model, demo_data, path, size = 10000, skip_fit = False):
    print
    print
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-------- re-train generator's decoder with sampled demo+random seed ----------"
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print
    print

    seeds = gen_seeds(size)
    losses = np.ones(size).reshape(-1,1)

    random_data = [seeds , losses]
    affine_data = [ demo_data[:,0:1,:], np.ones(demo_data.shape[0]).reshape(-1,1) ]

    sampled_data = merge_data(random_data, affine_data)

    config_generator (gene_model, deco_trainable = True, disc_trainable = False)
    if not skip_fit:
        gene_model = fit_and_save_model(gene_model, train = sampled_data, test = None, model_output_path = path, save_model = False, epochs = args.epochs, visualize_train = False, visualize_pred = False)

    gen_predicted_data = predict_scripts(gene_model, seeds)

    return gene_model, gen_predicted_data

def translate_and_merge(affine_train, gen_predicted_data, train_data):
    __MULTI_FACTOR__ = 1
    gen_X = gen_predicted_data[0]
    gen_Y = gen_predicted_data[1]

    translation_X = []
    translation_Y = []
    for index in range(gen_X.shape[0]):
        X = np.copy(gen_X[index])
        Y = gen_Y[index]
        translation_X.append(X)
        translation_Y.append(Y/3.0)

        dest = X[0,:]
        #visualize_script(X, dist = Y)

        for cnt in range(__MULTI_FACTOR__):
            into_index = np.random.randint(affine_train.shape[0])

            X = np.copy(affine_train[into_index])

            source = X[0,:]

            affine_X = X
            affine_X -= source
            affine_X += dest
            #visualize_script(affine_X, dist = "translated")
            affine_X = np.copy(affine_X)
    
            translation_X.append(affine_X)
            translation_Y.append(1.0)

    translation_X = np.array(translation_X)
    translation_Y = np.array(translation_Y).reshape(-1,1)

    translation_data = [ translation_X, translation_Y ]

    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!! train_data"
    print train_data[0].shape
    print "!!!!!!!!!!!!!! translation_data"
    print translation_data[0].shape
    data = merge_data(train_data, translation_data)
    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!!"
    print 
    print data[0].shape
    print data[1].shape

    accu_dataset_name_prefix = args.output_dir + "/temp_"

    accu_dataset_X = load_from_npy(accu_dataset_name_prefix + "X.npy")
    accu_dataset_Y = load_from_npy(accu_dataset_name_prefix + "Y.npy")


    size = args.__MAX_ACCU_DATA_SIZE__
    size = translation_data[0].shape[0] * 2
    if accu_dataset_X is not None:
        print "!!!!!!!!!!!!!!"
        print 
        print "!!!!!!!!!!!!!!"
        print 
        print "!!!!!!!!!!!!!!"
        print 
        print "!!!!!!!!!!!!!!"
        print 
        print accu_dataset_X.shape
        print accu_dataset_Y.shape

        accu_dataset = [accu_dataset_X, accu_dataset_Y]
        data = merge_data(data, accu_dataset)
        data = sample_from(data, size)

    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!!"
    print 
    print "!!!!!!!!!!!!!!"
    print 
    print data[0].shape
    print data[1].shape
    save_to_npy(data[0], accu_dataset_name_prefix + "X.npy")
    save_to_npy(data[1], accu_dataset_name_prefix + "Y.npy")

    for cnt in range(25):
        index = np.random.randint(data[0].shape[0])
        #visualize_script(data[0][index], dist = data[1][index])

    return data

def retrain_discriminator(affine_train, train_data, gen_predicted_data, test_data, path, size = 10000):
    print
    print
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-------- re-train standalone discriminator with sampled demo+gene_predicted seed ----------"
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print 

    print "-------- load_discriminator  ----------"
    disc_pack = load_model_pack(path = args.output_dir + args.__DISC_FOLDER__, model_name = args.__DISC_MODEL_NAME__, trainable = True, entry = args.__DISC_MODEL_NAME__ + "_entry", exit = args.__DISC_MODEL_NAME__ + "_exit")
    disc_model = disc_pack[2]

    sampled_data = translate_and_merge(affine_train, gen_predicted_data, train_data)

    disc_model = fit_and_save_model(disc_model, train = sampled_data, test = test_data, model_output_path = path, save_model = True, epochs = args.epochs, visualize_train = False, visualize_pred = False)

    return disc_model

def predict_scripts(model, seeds):
    [losses, scripts] = model.predict(seeds)

    scripts = scripts.reshape(seeds.shape[0], 64, 6)

    data = [scripts, losses]
    return data

def merge_data(data1, data2):
    sample = [
        np.concatenate([data1[0], data2[0]], axis = 0),
        np.concatenate([data1[1], data2[1]], axis = 0)
    ]
    return sample

def sample_from(data, size):
    X, Y = data
    sample_indices = np.random.choice(X.shape[0], size, replace=True)
    X = X[sample_indices]
    Y = Y[sample_indices]
    return [X, Y]

def save_sample_figures(scripts, save_path, prefix = "sample_fig", size = args.sive_fig_num):
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

def run_iteration(train_data, test_data, affine_train, affine_test, iteration = 0, size = 10000, path = args.__GENE_FOLDER__, save_samples_path = args.__FIG_FOLDER__):
    print 
    print 
    print 
    print "\n\n-------- Running Iteration %d  ----------\n\n" % iteration
    #need to load model every iteration to workaround what seems to be a tensorflow mem leak

    new_model_name = args.__DECO_MODEL_NAME__ + "_of_gene"
    print "\n\n-------- extracting_(%r)_from_generator  ----------\n\n" % new_model_name
    #deco_pack = extract_model_from_generator(model_name = args.__DECO_MODEL_NAME__, new_model_name = new_model_name)
    deco_pack = load_model_pack (path = args.output_dir + args.__DECO_FOLDER__, model_name = args.__DECO_MODEL_NAME__, entry = args.__DECO_MODEL_NAME__ + "_entry", exit = args.__DECO_MODEL_NAME__ + "_exit") 

    print "\n\n-------- load_discriminator  ----------\n\n"
    disc_pack = load_model_pack(path = args.output_dir + args.__DISC_FOLDER__, model_name = args.__DISC_MODEL_NAME__, entry = args.__DISC_MODEL_NAME__ + "_entry", exit = args.__DISC_MODEL_NAME__ + "_exit")

    gene_model = create_generator( decoder_pack = deco_pack, discriminator_pack = disc_pack )

    if iteration == 1:
        print gene_model.summary()

        prefix = "Demonstration"
        demo = affine_test
        #demo = affine_train[:1000] #TODO
        save_sample_figures(demo, save_samples_path, prefix)

    gene_model, gen_predicted_data = fit_generator_decoder(gene_model = gene_model, demo_data = affine_train, path = path, size = size, skip_fit = False)

    print "\n\n-------- predicting using seeds from affine_test data  ----------\n\n"
    prefix = "Predicted_In_Iteration_" + str(iteration)
    affine_test_seeds = affine_test[:,:1,:]
    #affine_test_seeds = affine_train[:1000,:1,:] #TODO
    gen_predicted_affine_test_data = predict_scripts(gene_model, affine_test_seeds)

    print "\n\n-------- saving samples of prediction to disk  ----------\n\n"
    save_sample_figures(gen_predicted_affine_test_data[0], save_samples_path, prefix)

    #gen_predicted_data != gen_predicted_affine_test_data
    retrain_discriminator(affine_train = affine_train, train_data = train_data, gen_predicted_data = gen_predicted_data, test_data = test_data, path = args.output_dir + args.__DISC_FOLDER__, size = size)

    #prefix = "posdisc_iter_" + str(iteration)

    return gene_model

def main():
    print "************ Running Experiement Iteration %d, re_train == %r ************\n\n\n" % (args.itera, args.re_train)
    
    print "\n\n-------- loading train/test dataset  ----------\n\n"
    test, train, affine_test, affine_train = load_np_data(args.output_dir, max_size = args.max_size, visualize = False)
    print "\ttrain size %d, affine_train size %d" % (train[0].shape[0], affine_train[0].shape[0])

    if args.re_train:
        print "\n\n-------- continue training generator  ----------\n\n"
        gene_model = run_iteration(train_data = train, test_data = test, affine_train = affine_train, affine_test = affine_test, iteration = args.itera, path = args.output_dir + args.__GENE_FOLDER__, size = args.max_size )

    else:
        print "\n\n-------- create_and_fit_discriminator  ----------\n\n"
        create_and_fit_discriminator(train = train, test = test, path = args.output_dir + args.__DISC_FOLDER__)

        print "\n\n-------- create_and_fit_decoder  ----------\n\n"
        create_and_fit_decoder (affine_train = affine_train, affine_test = affine_test, path = args.output_dir + args.__DECO_FOLDER__)

        print "\n\n-------- load_decoder  ----------\n\n"
        deco_pack = load_model_pack (path = args.output_dir + args.__DECO_FOLDER__, model_name = args.__DECO_MODEL_NAME__, entry = args.__DECO_MODEL_NAME__ + "_entry", exit = args.__DECO_MODEL_NAME__ + "_exit") 

        print "\n\n-------- load_discriminator  ----------\n\n"
        disc_pack = load_model_pack(path = args.output_dir + args.__DISC_FOLDER__, model_name = args.__DISC_MODEL_NAME__, entry = args.__DISC_MODEL_NAME__ + "_entry", exit = args.__DISC_MODEL_NAME__ + "_exit")

        print "\n\n-------- create_generator  ----------\n\n"
        gene_model = create_generator( decoder_pack = deco_pack, discriminator_pack = disc_pack )

        config_generator (gene_model, deco_trainable = False, disc_trainable = False)
        # only train a tiny dataset, and save the model
        mini_size = 3
        mini_data = [ np.ones(mini_size * 6).reshape(mini_size, 1, 6), np.ones(mini_size).reshape(-1, 1) ]

        gene_model = fit_and_save_model(gene_model, train = mini_data, test = None, model_output_path = args.output_dir + args.__GENE_FOLDER__, epochs = 1, visualize_train = False)
        

    return gene_model
    
if __name__== "__main__":
    args = argumentParser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()


'''
config_generator (gene_model, deco_trainable = False, disc_trainable = False)

'''
