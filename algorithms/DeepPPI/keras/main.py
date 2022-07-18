import os
import sys
import datetime
import argparse
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import data_loader as dl
# import mail_callback
from models.lstm32_3conv3_2dense import LSTM32_3Conv3_2Dense
from models.lstm32_2conv3_2dense_shared import LSTM32_2Conv3_2Dense_S
from models.lstm32_2conv3_4dense_shared import LSTM32_2Conv3_4Dense_S
from models.lstm32_3conv3_2dense_shared import LSTM32_3Conv3_2Dense_S
from models.lstm32_3conv4_2dense_shared import LSTM32_3Conv4_2Dense_S
from models.lstm32_3conv3_3dense_shared import LSTM32_3Conv3_3Dense_S
from models.lstm64_3conv3_2dense_shared import LSTM64_3Conv3_2Dense_S
from models.lstm64drop_3conv3_3dense_shared import LSTM64Drop_3Conv3_3Dense_S
from models.lstm64x2_3conv3_10dense_shared import LSTM64x2_3Conv3_10Dense_S
from models.lstm64x2_embed2_10dense_shared import LSTM64x2_Embed2_10Dense_S
from models.lstm64x2_embed4_10dense_shared import LSTM64x2_Embed4_10Dense_S
from models.fc6_embed3_2dense import FC6_Embed3_2Dense
from models.fc2_2dense import FC2_2Dense
from models.fc2_100_2dense import FC2_100_2Dense
from models.fc2_20_2dense import FC2_20_2Dense
from models.fc2_2_2dense import FC2_2_2Dense
from models.conv3_3_2dense_shared import Conv3_3_2Dense_S

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers as opti
from tensorflow.keras.utils import plot_model


# from tensorflow.keras.utils import multi_gpu_model
# import tensorflow.keras.backend.tensorflow_backend as KTF

def usage():
    print(
        "Usage: {} [train_set OR load_weights + test_set] <OPTIONS>\nEnter {} -h to have the list of optional arguments".format(
            sys.argv[0], sys.argv[0]))
    sys.exit(1)


## From https://stackoverflow.com/a/43357954/2007142
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def metrics(y_true, y_pred):
    # Count positive samples.
    diff = y_true + y_pred - 1
    true_positive = sum(diff == 1)
    pred_positive = sum(y_pred == 1)
    real_positive = sum(y_true == 1)

    # print('TP={}, pred pos={}, real pos={}'.format(true_positive, pred_positive, real_positive))

    # If there are no true samples, fix the F1 score at 0.
    if real_positive == 0:
        return 0

    # How many selected items are relevant?
    precision = true_positive / pred_positive

    # How many relevant items are selected?
    recall = true_positive / real_positive

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


# This factory also returns a string with the model name mainly to avoid
# the situation where we miswrite a model name and we are applying the
# default fc_flat without being aware of it
def factory_model(model_name):
    if model_name == 'lstm32_3conv3_2dense':
        return LSTM32_3Conv3_2Dense(), 'lstm32_3conv3_2dense'
    elif model_name == 'lstm32_2conv3_2dense_shared':
        return LSTM32_2Conv3_2Dense_S(), 'lstm32_2conv3_2dense_shared'
    elif model_name == 'lstm32_2conv3_4dense_shared':
        return LSTM32_2Conv3_4Dense_S(), 'lstm32_2conv3_4dense_shared'
    elif model_name == 'lstm32_3conv3_2dense_shared':
        return LSTM32_3Conv3_2Dense_S(), 'lstm32_3conv3_2dense_shared'
    elif model_name == 'lstm32_3conv4_2dense_shared':
        return LSTM32_3Conv4_2Dense_S(), 'lstm32_3conv4_2dense_shared'
    elif model_name == 'lstm32_3conv3_3dense_shared':
        return LSTM32_3Conv3_3Dense_S(), 'lstm32_3conv3_3dense_shared'
    elif model_name == 'lstm64_3conv3_2dense_shared':
        return LSTM64_3Conv3_2Dense_S(), 'lstm64_3conv3_2dense_shared'
    elif model_name == 'lstm64drop_3conv3_3dense_shared':
        return LSTM64Drop_3Conv3_3Dense_S(), 'lstm64drop_3conv3_3dense_shared'
    elif model_name == 'lstm64x2_3conv3_10dense_shared':
        return LSTM64x2_3Conv3_10Dense_S(), 'lstm64x2_3conv3_10dense_shared'
    elif model_name == 'lstm64x2_embed2_10dense_shared':
        return LSTM64x2_Embed2_10Dense_S(), 'lstm64x2_embed2_10dense_shared'
    elif model_name == 'lstm64x2_embed4_10dense_shared':
        return LSTM64x2_Embed4_10Dense_S(), 'lstm64x2_embed4_10dense_shared'
    elif model_name == 'fc6_embed3_2dense':
        return FC6_Embed3_2Dense(), 'fc6_embed3_2dense'
    elif model_name == 'fc2_2dense':
        return FC2_2Dense(), 'fc2_2dense'
    elif model_name == 'fc2_100_2dense':
        return FC2_100_2Dense(), 'fc2_100_2dense'
    elif model_name == 'fc2_20_2dense':
        return FC2_20_2Dense(), 'fc2_20_2dense'
    elif model_name == 'fc2_2_2dense':
        return FC2_2_2Dense(), 'fc2_2_2dense'
    elif model_name == 'conv3_3_2dense_shared':
        return Conv3_3_2Dense_S(), 'conv3_3_2dense_shared'
    else:
        print("Model unknown. Terminating.")
        sys.exit(1)


# This factory also returns a string with the optimizer name mainly to avoid
# the situation where we miswrite a optimizer name and we are applying the
# default adam without being aware of it
def factory_optimizer(optimizer_name, lr=0.001):
    if optimizer_name == 'sgd':
        return opti.SGD(learning_rate=lr), 'sgd'
    elif optimizer_name == 'rmsprop':
        return opti.RMSprop(learning_rate=lr), 'rmsprop'
    elif optimizer_name == 'adagrad':
        return opti.Adagrad(learning_rate=lr), 'adagrad'
    elif optimizer_name == 'adadelta':
        return opti.Adadelta(learning_rate=lr), 'adadelta'
    elif optimizer_name == 'adamax':
        return opti.Adamax(learning_rate=lr), 'adamax'
    elif optimizer_name == 'nadam':
        return opti.Nadam(learning_rate=lr), 'nadam'
    else:
        return opti.Adam(learning_rate=lr), 'adam'


def make_parser():
    '''
    Parsing function for the training and validation of networks
    '''
    parser = argparse.ArgumentParser(description='Protein-Protein interaction predicter')
    parser.add_argument('-train', type=str, help='File containing the training set')
    parser.add_argument('-val', type=str, help='File containing the validation set')
    parser.add_argument('-test', type=str, help='File containing the test set')
    parser.add_argument('-model', type=str,
                        help='choose among: lstm32_3conv3_2dense, lstm32_2conv3_2dense_shared, lstm32_3conv3_2dense_shared, lstm32_2conv3_4dense_shared, lstm32_3conv4_2dense_shared, lstm64_3conv3_2dense_shared, lstm64drop_3conv3_3dense_shared, lstm64x2_3conv3_10dense_shared, lstm64x2_embed2_10dense_shared, lstm64x2_embed4_10dense_shared, fc6_embed3_2dense, fc2_2dense, fc2_100_2dense, fc2_20_2dense, fc2_2_2dense, conv3_3_2dense_shared')
    parser.add_argument('-epochs', type=int, default=50, help='Number of epochs [default: 50]')
    parser.add_argument('-batch', type=int, default=64, help='Batch size [default: 64]')
    parser.add_argument('-patience', type=int, default=0,
                        help='Number of epochs before triggering the early stopping criterion [default: infinite patience]')
    parser.add_argument('-optimizer', type=str, default='adam',
                        help='Choose among: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam [default: adam]')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate [default: 0.001]')
    parser.add_argument('-gpu', type=int, default=0, help='If you have several GPUs, which one to use [default: 0]')
    parser.add_argument('-nb_gpu', type=int, default=1,
                        help='Number of GPU devices to use. Incompatible with the -gpu option [default: 1]')
    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                        help='To save weights of your model')
    parser.add_argument('-tensorboard', type=str2bool, nargs='?', const=True, default=False,
                        help='Save logs for TensorBoard')
    # parser.add_argument('-mail', type=str2bool, nargs='?', const=True, default=False, help='To automatically send an e-mail once training is over (private_mail_data.txt must be set properly)')
    parser.add_argument('-load', type=str,
                        help='File containing weights to load. You must also give a test set with this option.')
    parser.add_argument('-name', type=str,
                        help='Name complement to produced files, written at the end of the name file.')
    return parser


if __name__ == '__main__':

    # To make sure TS is only booking the right amount of GPU memory, instead of all memory available
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    parser = make_parser()
    args = parser.parse_args()
    model_name = args.model
    epochs = int(args.epochs)
    number_gpu = int(args.nb_gpu)
    which_gpu = '/gpu:' + str(args.gpu)
    batch_size = int(args.batch) * number_gpu
    patience = args.patience
    optimizer_name = args.optimizer
    lr = args.lr

    save_weights = bool(args.save)
    tensorboard = bool(args.tensorboard)
    # send_mail = bool(args.mail)
    load_weights = args.load
    name_complement = args.name

    train_set = args.train
    validation_set = args.val
    test_set = args.test

    if int(patience) == 0:
        patience = args.epochs

    if not train_set and not (test_set and load_weights):
        usage()

    # Result files will be saved using a name starting with file_name
    now = datetime.datetime.now()
    file_name = model_name + '_' + now.strftime("%Y-%m-%d_%H:%M") + '_gpu-' + str(args.gpu) + '-' + str(
        number_gpu) + '_' + optimizer_name + '_' + str(lr) + '_' + str(batch_size) + '_' + str(epochs)
    if name_complement:
        file_name = file_name + '_' + str(name_complement)

    if not load_weights:
        print("Loading training data")
        if "embed" in model_name:
            protein1, protein2, labels = dl.load_data_embed(train_set)
        else:
            protein1, protein2, labels = dl.load_data(train_set)
        print(f'{len(labels)} protein pairs in training!')

        if validation_set:
            print("Loading validation data")
            if "embed" in model_name:
                val_p1, val_p2, val_labels = dl.load_data_embed(validation_set)
            else:
                val_p1, val_p2, val_labels = dl.load_data(validation_set)
            print(f'{len(val_labels)} protein pairs in validation ...')

            if test_set:
                # callbacks_list = []
                callbacks_list = [
                    callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, min_lr=0.0008, cooldown=1,
                                                verbose=1)]
                callbacks_list.append(callbacks.EarlyStopping(monitor='acc', patience=patience, verbose=1))
                if save_weights:
                    callbacks_list.append(callbacks.ModelCheckpoint(filepath='weights/' + file_name + '.h5',
                                                                    monitor='loss', save_best_only=True, verbose=1))
            else:
                # callbacks_list = []
                callbacks_list = [
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0008, cooldown=1,
                                                verbose=1)]
                callbacks_list.append(callbacks.EarlyStopping(monitor='val_acc', patience=patience, verbose=1))
                # callbacks_list = [ callbacks.EarlyStopping(monitor='val_loss', patience=patience,) ]
                if save_weights:
                    callbacks_list.append(callbacks.ModelCheckpoint(filepath='weights/' + file_name + '.h5',
                                                                    monitor='val_loss', save_best_only=True, verbose=1))
        else:
            val_p1, val_p2, val_labels = None, None, None
            callbacks_list = [
                callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, min_lr=0.0008, cooldown=1,
                                            verbose=1)]
            callbacks_list.append(callbacks.EarlyStopping(monitor='acc', patience=patience, verbose=1))
            if save_weights:
                callbacks_list.append(callbacks.ModelCheckpoint(filepath='weights/' + file_name + '.h5',
                                                                monitor='loss', save_best_only=True, verbose=1))

        if tensorboard:
            callbacks_list.append(
                callbacks.TensorBoard(log_dir='./tensorboard_logs', histogram_freq=10, batch_size=batch_size,
                                      write_graph=False, write_grads=True, write_images=True, embeddings_freq=0,
                                      embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                      update_freq='epoch'))

    # if number_gpu > 1:
    #    print("Working on {} GPUs".format(number_gpu))
    #    with tf.device( "/cpu:0" ):
    #        # Build one model among available ones
    #        abstract_model, model_name = factory_model( model_name )
    #        model = abstract_model.get_model()
    #    model = multi_gpu_model(model, gpus=number_gpu)
    # else:
    with tf.device(which_gpu):
        # Build one model among available ones
        abstract_model, model_name = factory_model(model_name)
        model = abstract_model.get_model()

    optimizer, optimizer_name = factory_optimizer(optimizer_name, lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    # os.remove( 'models/figs/' + model_name + '_summary.txt' )
    with open('models/figs/' + model_name + '_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Draw our model in a png file
    # os.remove( 'models/figs/' + model_name + '.png' )
    # plot_model(model, show_shapes=True, to_file='models/figs/' + model_name + '.png')

    if not validation_set and test_set:
        print("Training model")
        train_data = [protein1, protein2]
        test_p1, test_p2, test_labels = dl.load_data(test_set)
        test_data = [test_p1, test_p2]
        print(f'{len(test_labels)} protein pairs in test!')
    elif not test_set:
        print("Training model")
        train_data = [protein1, protein2]
        val_data = ([val_p1, val_p2], val_labels)
    else:
        if not load_weights:
            print("Retraining on train+validation sets and testing on the test set")
            train_data = [np.concatenate([protein1, val_p1]), np.concatenate([protein2, val_p2])]
            labels = np.concatenate([labels, val_labels])
            val_data = None
        if "embed" in model_name:
            test_p1, test_p2, test_labels = dl.load_data_embed(test_set)
        else:
            test_p1, test_p2, test_labels = dl.load_data(test_set)
        test_data = [test_p1, test_p2]
        print(f'{len(test_labels)} protein pairs in test!')

    # os.remove( 'results/' + file_name + '_loss.png' )
    # os.remove( 'results/' + file_name + '_acc.png' )

    # String of arguments
    arguments = model_name + ', epochs=' + str(epochs) + ', batch=' + str(
        batch_size) + ', optimizer=' + optimizer_name + ', learning rate=' + str(lr) + ', patience=' + str(patience)

    result_file = open('results2/' + file_name + '.txt', 'w')
    print(f'Writing to results2/{file_name}.txt')
    result_file.write("File {}.txt\n".format(file_name))
    result_file.write(arguments + '\n')

    if load_weights:
        model.load_weights(load_weights)
    else:
        result_file.write('Number of training samples: {}\n\n'.format(len(labels)))
        if validation_set:
            history = model.fit(train_data,
                                labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=callbacks_list,
                                validation_data=val_data
                                )
        else:
            history = model.fit(train_data,
                                labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=callbacks_list,
                                validation_split=0.2
                                #validation_data=val_data
                                )

        # if save_weights:
        #     model.save_weights('weights/' + file_name + '.h5')

    if not test_set:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        result_file.write('Loss\n')
        for i in range(0, len(loss)):
            result_file.write('{}: train_loss={}, val_loss={}\n'.format(i, loss[i], val_loss[i]))

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.suptitle('Training and validation losses')
        plt.title(arguments)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        print(f'Saving to results2/{file_name}_loss.png')
        plt.savefig('results2/' + file_name + '_loss.png', bbox_inches='tight')

        plt.clf()

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        result_file.write('\n///////////////////////////////////////////\n\n')
        result_file.write('Accuracy\n')
        for i in range(0, len(loss)):
            result_file.write('{}: train_acc={}, val_acc={}\n'.format(i, acc[i], val_acc[i]))

        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
        plt.suptitle('Training and validation accuracies')
        plt.title(arguments)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        print(f'Saving to results2/{file_name}_acc.png')
        plt.savefig('results2/' + file_name + '_acc.png', bbox_inches='tight')

        result_file.write('\n///////////////////////////////////////////\n\n')
        result_file.write('Validation metrics\n')
        predict = model.predict([val_p1, val_p2], batch_size=batch_size, verbose=1)
        predict = np.reshape(predict, -1)
        y_pred = np.round(predict).astype(np.int8)
        pred_zero = np.count_nonzero(y_pred == 0)
        pred_one = np.count_nonzero(y_pred == 1)
        precision, recall, f1_score = metrics(y_pred, val_labels)
        print('\nNumber of 0 predicted: {}\nNumber of 1 predicted: {}\n'.format(pred_zero, pred_one))
        result_file.write('\nNumber of 0 predicted: {}\nNumber of 1 predicted: {}\n'.format(pred_zero, pred_one))
        print('Validation precision: {}\nValidation recall: {}\nValidation F1-score: {}\n\n'.format(precision, recall,
                                                                                                    f1_score))
        result_file.write(
            'Validation precision: {}\nValidation recall: {}\nValidation F1-score: {}\n\n'.format(precision, recall,
                                                                                                  f1_score))
    else:
        score, acc = model.evaluate(test_data, test_labels)
        print('Test loss: {}\nTest accuracy: {}'.format(score, acc))
        result_file.write('Test loss: {}\nTest accuracy: {}\n'.format(score, acc))
        predict = model.predict(test_data, batch_size=batch_size, verbose=1)
        predict = np.reshape(predict, -1)
        y_pred = np.round(predict).astype(np.int8)
        pred_zero = np.count_nonzero(y_pred == 0)
        pred_one = np.count_nonzero(y_pred == 1)
        precision, recall, f1_score = metrics(y_pred, test_labels)
        print('\nNumber of 0 predicted: {}\nNumber of 1 predicted: {}\n'.format(pred_zero, pred_one))
        result_file.write('\nNumber of 0 predicted: {}\nNumber of 1 predicted: {}\n'.format(pred_zero, pred_one))
        print('Test precision: {}\nTest recall: {}\nTest F1-score: {}\n\n'.format(precision, recall, f1_score))
        result_file.write(
            'Test precision: {}\nTest recall: {}\nTest F1-score: {}\n\n'.format(precision, recall, f1_score))

    result_file.close()
