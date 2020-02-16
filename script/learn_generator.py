from datetime import datetime

import gc

from process_scripts import *

import sys, datetime

from tensorflow import keras

from arg_parser import *

from keras import regularizers


#adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.999, beta_2=0.9999, amsgrad=False)

class IM_GAN():
    def __init__(self, gene_model, disc_model):
        __NAME__ = args.__GAN_MODEL_NAME__
        __NAME_PREFIX__ = __NAME__ + "_"

        self.dims = 6

        self.opti = keras.optimizers.Adam(0.0002, 0.5)


        self.discriminator = disc_model
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opti,
                                   metrics=['accuracy'])

        if args.state_to_state:
            in_shape = (args.state_history,6)
        else:
            in_shape = (1,6)
        seed = keras.layers.Input(shape=(in_shape), name = __NAME_PREFIX__ + "entry")
        self.generator = gene_model

        script = self.generator(seed)

        frozen = keras.Model(self.discriminator.inputs, self.discriminator.outputs)
        frozen.trainable = False
        loss = frozen(script)

        self.combined = keras.Model(seed, loss)
        loss = 'binary_crossentropy'
        self.combined.compile(loss=loss,
                              optimizer=self.opti,
                              metrics=['accuracy'])
        
        keras.utils.plot_model(
            self.combined,
            to_file=self.combined.name + '.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
        return

    def predict(self, seeds):
        pred = self.generator.predict(seeds)
        pred = pred.reshape(seeds.shape[0], -1, 6)
        return pred

    def predict_scripts(self, seeds):
        if args.state_to_state:
            scripts = infer_traj(self.generator, seeds)
        else:
            scripts = self.generator.predict(seeds)
            scripts = scripts.reshape(seeds.shape[0], 64, 6)

        return scripts

class State_To_State:
    def __init__(self):
        self.history = args.state_history

    def extract_state_to_state(self, trajs):
        data_X = []
        data_Y = []

        for traj in trajs:
            for index in range(len(traj)):
                X = []
                Y = []
                current = index - self.history
                for i in range(self.history):
                    if current + i < 0:
                        X.append ( traj[0] )
                    else:
                        X.append ( traj[current + i] )

                X = np.array(X)
                prev = X[:-1].mean(axis = 0)
                next = X[-1]
                diff = ((prev - next) ** 2).mean()
                if diff < 0.01:
                    continue
                else:
                    pass
                if index < len(traj) - 1:
                    if_exit = 0
                else:
                    if_exit = 1
                Y.append ( np.append(traj[index], if_exit) )
                data_X.append(X)
                data_Y.append(Y)

        data_X = np.array(data_X)
        data_Y = np.array(data_Y)

        if args.state_to_state:
            return np.concatenate( [ data_X[:,:,:6], data_Y[:,:,:6] ], axis = 1 )
        else:
            return data_X, data_Y
    

class IM_BC():
    def __init__(self):
        self.history = args.state_history
        self.model = HW_BC_feedforward(self.history)

        self.opti = keras.optimizers.Adam(0.0002, 0.5)
        loss = 'MSE'
        self.model.compile(loss=loss,
                           optimizer=self.opti,
                           metrics=['MAE'])
        
        keras.utils.plot_model(
            self.model,
            to_file=self.model.name + '.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
    def extract_state_to_state(self, traj):
        sa = State_To_State()
        return sa.extract_state_to_state(traj)

def infer_traj(model, seeds, max_len = 64):

    trajs = []
    for seed in seeds:
        traj = [] + list(seed)
        # take the first args.state_history points
        seed = seed[:args.state_history]
        seed = seed.reshape( [1, args.state_history, -1] )

        Y = []
    
        x = seed
        for i in range(max_len - args.state_history):
            y = model.predict(x)

            if args.state_to_state:
                #output is shape (B,6,6)
                #take the last point
                traj.append ( y[0,-1,:6] )
                x = np.concatenate( [ x[:,1:], y[:,-1:,:6] ], axis = 1 )
            else:
                #output is shape (B,1,7)
                traj.append ( y[0,0,:6] )
                if y[0,0,6] > 0.5:
                    break
                x = np.concatenate( [ x[:,1:], y[:,:,:6] ], axis = 1 )
            #print traj

        trajs.append( np.array(traj) )
    trajs = np.array(trajs)
    return trajs

def min_MSE(centered_gts, preds):
    print "min_MSE:"
    print centered_gts.shape
    return np.array([np.array([((pred-pred[0]) - (gt))**2 for gt in centered_gts]).mean(1).mean(1).min() for pred in preds])

def extract_endpoints(Y):
    X = np.concatenate([Y[:,:1], Y[:,-1:]], axis = 1)
    return X

def create_and_fit_ftf(affine_train, affine_test, path):
    train = [ extract_endpoints(affine_train) , affine_train ]
    test = [ extract_endpoints(affine_test) , affine_test ]
    model = HW_full_traj_feedforward()
    #args.epochs = 1
    model = fit_and_save_model(model, train, test, path, epochs = args.epochs, visualize_pred = True)
    print model.summary()
    #answer = model.predict()
    return

def HW_full_traj_feedforward():
    __NAME__ = "Feedforward_Conv_"
    __NAME_PREFIX__ = __NAME__ + "_"
    
    entry = keras.layers.Input(shape=(2,6), name = __NAME_PREFIX__ + "entry")
    dense_shape = np.array([16,64,128,128,256])

    x = entry
    x = keras.layers.Reshape((2,6), name = __NAME_PREFIX__ + "_entry_reshape")(x)

    w = 2
    for j in range(len(dense_shape)):
        filters = dense_shape[j] * 2
        act = 'linear'
        act = 'tanh'
        conv = keras.layers.Conv1D(filters,
                                   filters,
                                   activation=act,
                                   padding = 'same',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   name = __NAME_PREFIX__+ 'cv1_1_%d_x_%d' %(filters, j) )
        activation= keras.layers.ReLU(max_value = 6,
                                      name = __NAME_PREFIX__+ 'act_1_%d_x_%d' %(filters, j) )
        x = conv(x)
        x = activation(x)
        x = keras.layers.Reshape((w,1,filters), name = __NAME_PREFIX__ + "%d_1_reshape"%j)(x)
        x = keras.backend.resize_images(x, 1, 2, "channels_last")
        w *= 2
        x = keras.layers.Reshape((w,filters), name = __NAME_PREFIX__ + "%d_2_reshape"%j)(x)

    activation = 'sigmoid'
    x = keras.layers.Dense(6, activation=activation, name = __NAME_PREFIX__ + "dense_last" )(x)

    exit = keras.layers.Reshape((64,6), name = __NAME_PREFIX__ + "exit")(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    #print model.summary()

    model.compile(loss='MSE',
                  optimizer='adam',
                  metrics=['MAE'])
        
    keras.utils.plot_model(
        model,
        to_file=model.name + '.png',
        show_shapes=False,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    return model

def HW_BC_feedforward():
    entry = keras.layers.Input(shape=(history_len,6), name = __NAME_PREFIX__ + "entry")
    dense_shape = np.array([64,128,1024,512,32])

    x = entry
    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    for j in range(len(dense_shape)):
        filters = dense_shape[j]
        activation = 'relu'
        x = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense_" + str(j) )(x)

    activation = 'sigmoid'
    x = keras.layers.Dense(7, activation=activation, name = __NAME_PREFIX__ + "dense_last" )(x)

    exit = keras.layers.Reshape((1,7), name = __NAME_PREFIX__ + "exit")(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()
    

def HW_Disc_Dense(input_shape):
    __NAME__ = args.__DISC_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"
 
    dense_shape = np.array([32,128,1024,512,32])
    #dense_shape = np.array([12,256,1024,512]) v1

    __FILTER_SIZE__ = 7
    

    def add_disc_repetition_unit_v1(x, ori_x, filters, j):
        x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten_%d" % (j))(x)
        x = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense_" + str(j) )(x)

        return x

    def add_disc_repetition_unit_v2(x, ori_x, filters, j):
        filter_size = __FILTER_SIZE__
        factor = 2

        x1 = keras.layers.Conv1D(filters, filter_size, activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'cv1_1_%d_x_%d' %(filters, j) )(x)

        input_x_size = x.shape[1]

        x3 = keras.layers.Dense(filters, activation=activation, name = __NAME_PREFIX__ + "dense_3_" + str(j) )(x)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' % (j))([x1, x3])

        x = keras.layers.MaxPooling1D(name = __NAME_PREFIX__+ 'maxpool_num_%d' % (j), pool_size = 2)(x)
        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)

        return x

    def add_disc_repetition_unit_v3(x, seed_x, filters, j):
        filter_size = __FILTER_SIZE__
        factor = 2

        activation= keras.layers.LeakyReLU(alpha=0.4,
                                           name = __NAME_PREFIX__+ 'act_1_%d_x_%d' %(filters, j) )

        ori_x = x
        
        x = keras.layers.Conv1D(filters,
                                filter_size, activation='linear',
                                padding = 'same',
                                kernel_regularizer=regularizers.l2(0.001),
                                #activity_regularizer=regularizers.l1(0.0001),
                                name = __NAME_PREFIX__+ 'cv1_1_%d_x_%d' %(filters, j) )(x)
        
        x = activation(x)

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
        x = add_disc_repetition_unit_v3(x, ori_x, filters, j)

    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    activation= keras.layers.LeakyReLU(alpha=0.2,
                                       name = __NAME_PREFIX__+ 'act' )

    x = keras.layers.Dense(1, activation='linear', name = __NAME_PREFIX__ + "shift" )(x)

    x = activation(x)

    exit = keras.layers.Dense(1, activation='sigmoid', name = __NAME_PREFIX__ + "exit" )(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()

    return model

def HW_Disc_Dense_STS(input_shape):
    __NAME__ = args.__DISC_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"
 
    dense_shape = np.array([32,64,128,256,64,32])

    __FILTER_SIZE__ = 2
    

    def add_disc_repetition_unit_v3(x, seed_x, filters, j):
        filter_size = __FILTER_SIZE__
        factor = 2

        activation= keras.layers.LeakyReLU(alpha=0.4,
                                           name = __NAME_PREFIX__+ 'act_1_%d_x_%d' %(filters, j) )

        ori_x = x
        
        x = keras.layers.Conv1D(filters,
                                filter_size, activation='linear',
                                padding = 'same',
                                kernel_regularizer=regularizers.l2(0.001),
                                #activity_regularizer=regularizers.l1(0.0001),
                                name = __NAME_PREFIX__+ 'cv1_1_%d_x_%d' %(filters, j) )(x)
        
        x = activation(x)

        if j%3==0:
            x = keras.layers.MaxPooling1D(name = __NAME_PREFIX__+ 'maxpool_num_%d' % (j), pool_size = 2)(x)

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)

        return x


    entry = keras.layers.Input(shape=(args.state_history + 1, 6), name = __NAME_PREFIX__ + "entry")

    x = entry

    v1 = keras.layers.Cropping1D(cropping=([1, 0]), name = __NAME_PREFIX__ + "cropping_v1" )(x)
    v2 = keras.layers.Cropping1D(cropping=([0, 1]), name = __NAME_PREFIX__ + "cropping_v2" )(x)

    v = keras.layers.Subtract(name = __NAME_PREFIX__+ 'sub_v')([v2,v1])

    x = keras.layers.Concatenate(name = __NAME_PREFIX__+ 'concat_v', axis = 2)([v,v2])

    
    ori_x = x

    for j in range(len(dense_shape)):
        filters = dense_shape[j]
        x = add_disc_repetition_unit_v3(x, ori_x, filters, j)

    x = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten")(x)

    activation= keras.layers.LeakyReLU(alpha=0.2,
                                       name = __NAME_PREFIX__+ 'act' )

    x = keras.layers.Dense(1, activation='linear', name = __NAME_PREFIX__ + "shift" )(x)

    x = activation(x)

    exit = keras.layers.Dense(1, activation='sigmoid', name = __NAME_PREFIX__ + "exit" )(x)

    model = keras.Model(inputs=entry, outputs=exit, name= __NAME__)

    print model.summary()

    return model

def HW_Decoder():
    __NAME__ = args.__DECO_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 7

    __SEED_TRANS_NUM__ = 7

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
        x2 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv2_%d_num_%d' %(filters, j) )(x21)

        x31 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv31_%d_num_%d' %(filters, j) )(x)
        x32 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv32_%d_num_%d' %(filters, j) )(x31)
        x3 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv3_%d_num_%d' %(filters, j) )(x32)

        (b_size, x_size, y_size, c_size) = base_layer.output_shape

        x51 = keras.layers.Dense(c_size, activation=activation, name = __NAME_PREFIX__ + "seed51_%d_num_%d" %(filters, j) )(seed)
        x52 = keras.layers.Dense(c_size * x_size * y_size, activation=activation, name = __NAME_PREFIX__ + "seed52_%d_num_%d" %(filters, j) )(x51)
        x5 = keras.layers.Reshape([x_size, y_size, c_size])(x52)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' %(j))([x1, x2, x3, x5])

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)
        return x






    seed = keras.layers.Input(shape=[1,6], name = __NAME_PREFIX__ + "entry")

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



def HW_Decoder_STS():
    __NAME__ = args.__DECO_MODEL_NAME__
    __NAME_PREFIX__ = __NAME__ + "_"

    __FILTER_SIZE__ = 7

    __SEED_TRANS_NUM__ = 7

    __SHAPE__ = [32, 64, 128, 256]


    conv_shape = __SHAPE__
    deconv_shape = __SHAPE__
    filter_size = __FILTER_SIZE__



    def add_repetition_unit_v2(x, seed, filters, j):
        ori_x = x

        base_layer = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv1_%d_num_%d' %(filters, j) )
        x1 = base_layer(x)

        x21 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv21_%d_num_%d' %(filters, j) )(x)
        x2 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv2_%d_num_%d' %(filters, j) )(x21)

        x31 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv31_%d_num_%d' %(filters, j) )(x)
        x32 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, padding = 'same', name = __NAME_PREFIX__+ 'deconv32_%d_num_%d' %(filters, j) )(x31)
        x3 = keras.layers.Conv2DTranspose(filters, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv3_%d_num_%d' %(filters, j) )(x32)

        (b_size, x_size, y_size, c_size) = base_layer.output_shape

        x50 = keras.layers.Flatten(name = __NAME_PREFIX__ + "flatten_%d" % j)(x)
        x51 = keras.layers.Dense(5 * c_size, activation=activation, name = __NAME_PREFIX__ + "seed51_%d_num_%d" %(filters, j) )(x50)
        x52 = keras.layers.Dense(c_size * x_size * y_size, activation=activation, name = __NAME_PREFIX__ + "seed52_%d_num_%d" %(filters, j) )(x51)
        x5 = keras.layers.Reshape([x_size, y_size, c_size])(x52)

        x = keras.layers.Add(name = __NAME_PREFIX__+ 'add_num_%d' %(j))([x1, x2, x3])

        x = keras.layers.Dropout(name = __NAME_PREFIX__+ 'dropout_num_%d' %(j), rate = 0.2)(x)
        return x






    seed = keras.layers.Input(shape=[args.state_history,6], name = __NAME_PREFIX__ + "entry")

    stride = 1
    padding = 0
    activation = 'relu'
    activation = 'tanh'

    seed_trans_num = __SEED_TRANS_NUM__

    last_layer_x = args.state_history + 1

    seed2d = keras.layers.Reshape((args.state_history,1,6), name = __NAME_PREFIX__ + "seed2d")(seed)

    start = seed2d
    x = start
    for i in range(seed_trans_num):
        x = keras.layers.Dense(6, activation=activation, name = __NAME_PREFIX__ + "dense_start_num_%d" %(i) )(x)

    x = keras.layers.Conv2DTranspose(6, (last_layer_x, 1), activation='sigmoid', name = __NAME_PREFIX__+ 'deconv_first' )(x)

    for j, filters in enumerate(conv_shape):
        x = add_repetition_unit_v2(x, seed2d, filters, j)


    x = keras.layers.Conv2DTranspose(6, (filter_size, 1), activation=activation, name = __NAME_PREFIX__+ 'deconv_last_layer' )(x)

    for j in range(5):
        x = keras.layers.Conv2D(6, (filter_size,1), activation=activation, padding = 'valid', name = __NAME_PREFIX__+ 'downgrade_x_%d' %(j) )(x)
    x = keras.layers.AveragePooling2D((10,1), name = __NAME_PREFIX__+ 'downgrade_avg' )(x)

    x = keras.layers.Concatenate(axis = 1)([ start , x])

    x = keras.layers.Reshape((args.state_history + 1,6), name = args.__DECO_MODEL_NAME__ + "_exit")(x)

    exit = x

    model = keras.Model(inputs=seed, outputs=exit , name= __NAME__)

    print model.summary()

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

    os.system('cp ' + checkpoint + ' ' + checkpoint + '_back')

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint, verbose=1, save_best_only=True, mode='auto') #monitor='val_acc'

    log_dir = model_output_path + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=0.001, verbose=1)

    #callbacks = [cp_callback, tensorboard_callback, stopping_callback]
    callbacks = [cp_callback, tensorboard_callback]
    #callbacks = [tensorboard_callback, stopping_callback]

    batch = args.batch

    X, Y = train

    if visualize_train:
        for i in range(0, len(X), len(X)/37 + 1):
            x = X[i]
            y = Y[i]
            visualize_script(x, dist = y)

    history = model.fit(X, Y, epochs = epochs, batch_size= batch, shuffle=True, validation_split=( args.__TEST_RATIO__), callbacks=callbacks)

    val_loss = np.array(history.history['val_loss'])
    print "val_loss array: %r" % val_loss
    val_loss = val_loss.min()
    os.system('echo ' + str(val_loss) + ' >> ' + model_output_path + '/' + model.name + '_fit_history.txt')

    if (val_loss > 0.18):
        print "val_loss %r above threthold, overwriting with previous checkpoint" % val_loss
        os.system('cp ' + checkpoint + '_back' + ' ' + checkpoint)


    if save_model:
        model.save(model_output_path + model.name + '.h5')

    #model.save(model_output_path + model.name + '.tf', save_format="tf")
    if test:
        model.evaluate(test[0], test[1])

    if visualize_pred:
        answer = model.predict(X)

        if X.shape[1:] == [64,6]:
            #script = answer[1] # [loss, script]
            print "visualizing discri prediction"
            for cnt in range(35):#range(test[0].shape[0]):
                index = np.random.randint(X.shape[0])
                visualize_script(X[index], dist = "pred: %r" % answer[index][0])
        else:
            print "visualizing prediction"
            script = answer
            for cnt in range(X.shape[0]):
                #index = np.random.randint(X.shape[0])
                index= cnt
                filename = model_output_path + '/%02d_pred_sample.png' % cnt
                visualize_script(script[index], dist = "Prediction" , filename = filename)

                filename = model_output_path + '/%02d__ground_truth.png' % cnt
                visualize_script(Y[index], dist = "Ground Truth", filename = filename)
                if cnt > 100:
                    break
        
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

    os.system('cp ' + checkpoint + ' ' + checkpoint + '_back')
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

            step = max_size / 7
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

def create_decoder(input_shape):
    if args.state_to_state:
        model = HW_Decoder_STS()
        input_shape = (args.state_history, 6)
    else:
        model = HW_Decoder()
        
    model.compile(optimizer = 'adam',
                  loss='MSE',
                  metrics=['MAE']
    )

    model.build(input_shape)

    keras.utils.plot_model(
        model,
        to_file=model.name + '.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    return model


def create_discriminator(input_shape):
    if args.state_to_state:
        model = HW_Disc_Dense_STS(input_shape)
    else:
        model = HW_Disc_Dense(input_shape)
        
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']
    )

    model.build(input_shape)

    keras.utils.plot_model(
        model,
        to_file=model.name + '.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    
    return model

def create_and_fit_discriminator(affine_train, affine_test, train, test, path):

    if args.state_to_state:
        train = [
            np.zeros([args.batch, args.state_history + 1, 6]),
            np.zeros([args.batch, 1])
        ]
        test = None
    model = create_discriminator(train[0].shape)

    # pretrain with jump start data
    fit_and_save_model(model, train, test, path, epochs = args.epochs, visualize_train = False)
    return model


def find_entry_exit(model, entry = "entry", exit = "exit"):
    model_pack = [None, None, model]
    for layer in model.layers:
        if entry in layer.name:
            model_pack[0] = layer
        elif exit in layer.name:
            model_pack[1] = layer

    return model_pack


def extract_gene_from_gan(model_name, new_model_name):
    entry, exit, gan_model = load_model_pack (path = args.output_dir + args.__GAN_FOLDER__, model_name = args.__GAN_MODEL_NAME__)

    submodel = None
    for layer in gan_model.layers:
        if model_name == layer.name:
            submodel = layer

    if submodel is None:
        return None

    entry = keras.layers.Input(shape=(64,6), name = new_model_name + "_entry")

    x = entry

    exit = submodel(x)

    generator_model = keras.Model(inputs=entry, outputs=exit , name= new_model_name)

    generator_model.compile(optimizer='adam',
                  loss='MSE',
    )

    print "\n\n-------- extracted (%r) as:  ----------\n\n" % model_name
    print generator_model.summary()
    
    return [entry, exit, generator_model]



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

def rand_seeds(size = 10000):
    seeds = np.random.rand(size, 6)

    seeds [:,0] *= 0.6
    seeds [:,1] *= 0.3
    seeds [:,1] += 0.2
    seeds [:,2] *= 0.1
    seeds [:,2] += 0.1
    seeds [:,3] *= 0.1
    seeds [:,3] += 0.2
    seeds [:,4] *= 0.1
    seeds [:,5] *= 0.4

    return seeds.reshape([-1,1,6])

def create_and_fit_decoder(affine_train, affine_test, path):
    # don't pre fit
    size = 3
    if args.state_to_state:
        x = np.zeros([args.batch, args.state_history, 6])
        y = np.zeros([args.batch, args.state_history + 1, 6])
        affine_data_train = [x,y]
        affine_data_test = None
        deco_model = create_decoder(x.shape)

    else:
        affine_data_train = [affine_train[:size,0:1,:], affine_train[:size]]
        affine_data_test = None
        deco_model = create_decoder([None,1,6])
        
    fit_and_save_model(deco_model, affine_data_train, affine_data_test, path, epochs = 1, visualize_train = False, visualize_pred = False)

    return deco_model

def create_and_fit_bc(affine_train, affine_test, path):
    bc = IM_BC()

    train = bc.extract_state_to_state(affine_train)
    test = bc.extract_state_to_state(affine_test)

    fit_and_save_model(bc.model, train, test, path, epochs = args.epochs, visualize_train = False, visualize_pred = False)

    return bc.model
        
def fit_generator_from_gan(gan_model, demo_data, path, size = 10000, skip_fit = False):
    print
    print
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-------- re-train generator with sampled demo+random seed ----------"
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print "-----" * 20
    print
    print

    seeds = rand_seeds(size)
    losses = np.ones(size).reshape(-1,1)

    random_data = [seeds , losses]
    affine_data = [ demo_data[:,0:1,:], np.ones(demo_data.shape[0]).reshape(-1,1) ]

    sampled_data = merge_data(random_data, affine_data)

    config_gan (gan_model, deco_trainable = True, disc_trainable = False)
    if not skip_fit:
        gan_model = fit_and_save_model(gan_model, train = sampled_data, test = None, model_output_path = path, save_model = False, epochs = args.epochs, visualize_train = False, visualize_pred = False)

    gen_predicted_data = gan_model.predict_scripts(seeds)

    return gan_model, gen_predicted_data

def visualize(X, Y = None, size = 10, save_path = None, dequant = False):
    if len(X.shape) == 3:
        for cnt in range(size):
            index = np.random.randint(X.shape[0])
            if Y is None:
                y = None
            else:
                y = Y[index]
            if save_path is None:
                filename = None
            else:
                os.system('mkdir -p ' + save_path)
                filename = save_path + ('/vis_sample_%d.png' % cnt)
            visualize_script(X[index], y, filename = filename)
    elif len(X.shape) == 2:
        if Y is None:
            Y = None
        else:
            Y = Y
        visualize_script(X, Y)

        
def merge_translated(predicted_data, affine_train, prev_merged_data):
    __MULTI_FACTOR__ = 2
    X_arr = predicted_data[0]
    Y_arr = predicted_data[1]

    translation_X = []
    translation_Y = []
    for index in range(X_arr.shape[0]):
        X = np.copy(X_arr[index])
        Y = Y_arr[index]
        translation_X.append(X)
        translation_Y.append(Y)

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
    data = merge_data(translation_data, prev_merged_data)
    #visualize(data[0], data[1])
    return data
    
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
        translation_Y.append(0.)

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

    data = merge_data(train_data, translation_data)

    accu_dataset_name_prefix = args.output_dir + "/temp_"

    accu_dataset_X = load_from_npy(accu_dataset_name_prefix + "X.npy")
    accu_dataset_Y = load_from_npy(accu_dataset_name_prefix + "Y.npy")


    size = translation_data[0].shape[0] * 2
    if size > args.__MAX_ACCU_DATA_SIZE__:
        size = args.__MAX_ACCU_DATA_SIZE__

    if accu_dataset_X is not None:
        accu_dataset = [accu_dataset_X, accu_dataset_Y]
        data = merge_data(data, accu_dataset)
        data = sample_from(data, size)

    save_to_npy(data[0], accu_dataset_name_prefix + "X.npy")
    save_to_npy(data[1], accu_dataset_name_prefix + "Y.npy")

    for cnt in range(25):
        index = np.random.randint(data[0].shape[0])
        #visualize_script(data[0][index], dist = data[1][index])

    return data

def merge_data(data1, data2):
    sample = [
        np.concatenate([data1[0], data2[0]], axis = 0),
        np.concatenate([data1[1], data2[1]], axis = 0)
    ]
    return sample

def sample_from(data, size):
    X, Y = data
    if size > X.shape[0]:
        size = X.shape[0]
    sample_indices = np.random.choice(X.shape[0], size, replace=True)
    X = X[sample_indices]
    Y = Y[sample_indices]
    return [X, Y]

def save_sample_figures(scripts, save_path, prefix = "sample_fig", size = args.save_fig_num):
    max_size = 100;
    if save_path is not None:
        path = args.output_dir + "/" + save_path
        os.system('mkdir -p ' + path)

        total_size = scripts.shape[0]
        step = (total_size / args.save_fig_num) - 1
        if step < 1:
            step = 1
        for sample_index in range(0, total_size, step):
            script = scripts[sample_index]
            sample_script = np.copy(script).reshape([64,6])

            description = ("#%02d" % sample_index) + "_" + prefix
            filename = path + "/" + description
            visualize_script(sample_script, filename = filename, dist = description, dequant = True, write_traj = True)

            max_size -= 1
            if max_size == 0:
                break

def train_GAN(deco, disc, train, test, affine_test, affine_train, iterations = 0):
    print 
    print 
    print 
    print "\n\n-------- Running Iterations %d  ----------\n\n" % iterations

    print "\n\n-------- create_GAN  ----------\n\n"

    train_centered = np.array([traj - traj[0] for traj in affine_train])
    test_centered = np.array([traj - traj[0] for traj in affine_test])

    batch = args.batch
    if deco is None:
        deco = HW_Decoder()
    else:
        deco.trainable = True
        
    if disc is None:
        disc = HW_Disc_Dense()
    else:
        disc.trainable = True
        
    gan = IM_GAN(gene_model = deco, disc_model = disc)

    real = np.ones ((batch, 1))
    fake = np.zeros((batch, 1))

    save_freq = iterations / 100
    if save_freq < 1:
        save_freq = 2
    save_freq = 100 # overwrite TODO
    
    prefix = "Test_Demo"
    if not args.state_to_state:
        save_sample_figures(affine_test, args.save_fig_folder, prefix)

    for itera in range(iterations):
        x_train = train

        x_batch = x_train[np.random.randint(0, x_train.shape[0], batch)]

        if args.state_to_state:
            random_seeds = x_batch[:,:args.state_history,:]
        else:
            random_seeds = x_batch[:,:1,:]

        for j in range(args.epochs):
            if args.state_to_state:
                random_seeds = x_batch[:,:args.state_history,:]
            else:
                random_seeds = x_batch[:,:1,:]
            train_predicted = gan.predict(random_seeds)

            d_train_real = gan.discriminator.train_on_batch(x_batch, real)
            d_train_fake = gan.discriminator.train_on_batch(train_predicted, fake)
            d_train = (np.array(d_train_real) + np.array(d_train_fake))/2

        g_train = gan.combined.train_on_batch(random_seeds, real)
        for j in range(args.epochs*args.gen_steps):
            if itera % 2 == 0:
                random_seeds = rand_seeds(batch)
            else:
                random_seeds = x_batch[:,:1,:]
            if args.state_to_state:
                random_seeds = x_batch[:,:args.state_history,:]
            g_train = gan.combined.train_on_batch(random_seeds, real)

        log_file_name = args.output_dir + args.save_fig_folder + '/' + args.save_fig_name +'.txt'
        if itera == 0:
            header = '//itera' + ', d_train' + str(gan.discriminator.metrics_names) + ', g_train' + str(gan.combined.metrics_names) + ', d_eval' + str(gan.discriminator.metrics_names) + ', g_eval' + str(gan.combined.metrics_names) + ', traj_train_MSE' + ', traj_eval_MSE' + '//batch %d' % (batch)

            os.system('echo ' + header + ' > ' + log_file_name)

        if args.save_fig_folder and (itera % save_freq == 0):
            #print "\n\n-------- predicting test using seeds  ----------\n\n"

            if args.state_to_state:
                test_seeds = affine_test[:,:args.state_history,:]
            else:
                test_seeds = affine_test[:,:1,:]

            predicted_scripts  = gan.predict_scripts(test_seeds)

            traj_train_min_MSE = min_MSE(train_centered, np.copy(predicted_scripts)).mean(0)
            traj_eval_min_MSE = min_MSE(test_centered, np.copy(predicted_scripts)).mean(0)

            d_eval = gan.discriminator.evaluate(test, np.ones ((test.shape[0], 1)))
            g_eval = gan.combined.evaluate(test_seeds, np.ones ((test_seeds.shape[0], 1)))

            print "\n\n-------- saving samples of prediction to disk  ----------\n\n"
            prefix = "Test_Predicted_In_Iteration_" + str(itera)
            save_sample_figures(predicted_scripts, args.save_fig_folder, prefix)

            log = "%d \t %r \t %r \t %r \t %r \t %r \t %r \t %r \t %r \t %r %r" %(itera, d_train[0], d_train[1], g_train[0], g_train[1], d_eval[0], d_eval[1], g_eval[0], g_eval[1], traj_train_min_MSE, traj_eval_min_MSE)
            os.system('echo ' + log + ' >> ' + log_file_name)
            print header
            print log
            print datetime.datetime.now()

    #prefix = "posdisc_iter_" + str(iteration)

    return gan

def main():
    print "************ Running Experiement Iteration %d, train_gan == %r ************\n\n\n" % (args.itera, args.train_gan)
    
    print "\n\n-------- loading train/test dataset  ----------\n\n"
    test, train, affine_test, affine_train = load_np_data(args.output_dir, max_size = args.max_size, visualize = False)
    print "\ttrain size %d, affine_train size %d" % (train[0].shape[0], affine_train[0].shape[0])

    if args.train_gan:
        print "\n\n-------- load_decoder  ----------\n\n"
        deco_pack = load_model_pack (path = args.output_dir + args.__DECO_FOLDER__, model_name = args.__DECO_MODEL_NAME__, entry = args.__DECO_MODEL_NAME__ + "_entry", exit = args.__DECO_MODEL_NAME__ + "_exit") 

        print "\n\n-------- load_discriminator  ----------\n\n"
        disc_pack = load_model_pack(path = args.output_dir + args.__DISC_FOLDER__, model_name = args.__DISC_MODEL_NAME__, entry = args.__DISC_MODEL_NAME__ + "_entry", exit = args.__DISC_MODEL_NAME__ + "_exit")

        print "\n\n-------- train GAN  ----------\n\n"
        deco = deco_pack[2]
        disc = disc_pack[2]
        if args.state_to_state:
            sa = State_To_State()
            train = sa.extract_state_to_state(affine_train)
            test = sa.extract_state_to_state(affine_test)
            #ground_truth = np.concatenate([affine_train, affine_test], axis = 0)
            ground_truth = affine_test
        else:
            train = affine_train
            test = affine_test
            ground_truth = affine_test
        gan_obj = train_GAN(deco, disc, train = train, test = test, affine_test = affine_test, affine_train = affine_train, iterations = args.itera)

    elif args.init_gan:
        print "\n\n-------- create_and_fit_discriminator  ----------\n\n"
        create_and_fit_discriminator(affine_train = affine_train, affine_test = affine_test, train = train, test = test, path = args.output_dir + args.__DISC_FOLDER__)

        print "\n\n-------- create_and_fit_decoder  ----------\n\n"
        create_and_fit_decoder (affine_train = affine_train, affine_test = affine_test, path = args.output_dir + args.__DECO_FOLDER__)
    elif args.train_bc:
        bc_model = create_and_fit_bc (affine_train = affine_train, affine_test = affine_test, path = args.output_dir + args.__BC_FOLDER__)

        trajs = infer_traj(bc_model, affine_test)
        visualize(trajs, save_path = args.output_dir + args.save_fig_folder, dequant = True)

    return
    
if __name__== "__main__":
    args = argumentParser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()


