import argparse, os, csv, numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Input, Flatten, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras import optimizers
from random import shuffle
from PIL import Image
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from keras.losses import mean_squared_error
from sklearn.model_selection import cross_val_score
from keras.preprocessing.image import ImageDataGenerator


def generator(batch_size, data, label):
    filenames = get_filenames(label)
    counter = 0
    while 1:
        files_to_select = filenames[counter:counter+batch_size]
        X, y = data_read(data, label, files_to_select)
        counter += batch_size
        yield X, y

def data_read(data, label, filenames):
    x, y, file_index, features = [], [], [], []
    with open(label, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            file_index.append(row[0])
            features.append(row[1:])

    for file in filenames:
        input_path = os.path.join(data, file)
        im = np.asarray(Image.open(input_path))
        x.append(im)

        idx = file_index.index(file)
        y.append(features[idx])

    return np.array(x), np.array(y)


def get_filenames(label):
    with open(label, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        filenames = [f[0] for f in csv_reader]
    shuffle(filenames)
    return filenames


def train_model():
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3))


    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(8)(x)


    model = Model(inputs = model.input, outputs = predictions)
    for layer in model.layers[:5]:
        layer.trainable = False
    print(model.summary())
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=mean_squared_error,optimizer=adam)
    return model


def visualization(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig = plt.figure()
    fig.savefig('../visual/accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig_loss = plt.figure()
    fig_loss.savefig('../visual/loss.png')
    plt.show()

def train(args):
    steps = int(410 / args.batch_size)
    model = train_model()
    ckpt = args.ckpt + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(ckpt, monitor='val_loss', verbose=1, save_best_only=True)
    callback_list = [checkpoint]

    train_gen = generator(args.batch_size, args.train_data, args.train_label)
    val_gen = generator(args.batch_size, args.val_data, args.val_label)

    train_files = get_filenames(args.train_label)
    train_x, train_y = data_read(args.train_data, args.train_label, train_files)
    datagen = ImageDataGenerator()
    datagen.fit(train_x)
    val_filenames = get_filenames(args.val_label)
    val_X, val_y = data_read(args.val_data, args.val_label, val_filenames)

    history = model.fit_generator(train_gen, steps_per_epoch=steps, epochs=args.epochs, verbose=1,
                                  callbacks=callback_list, validation_data=val_gen,
                                    validation_steps=int(178/args.batch_size))
    '''
    estimator = KerasRegressor(build_fn=train_model(), epochs = 100, batch_size = args.batch_size, verbose = 1)
    results = cross_val_score(estimator=estimator, X=train_x, y=train_y)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    '''
    visualization(history=history)

def main():
    parser = argparse.ArgumentParser(description='It is for document localization')
    parser.add_argument('--train_data', default= '../data/train_image',
                        help='provide absolute path to input data')
    parser.add_argument('--train_label', default= '../data/coordinates/translated_coords.csv',
                        help = 'provide relative path to input labels')
    parser.add_argument('--val_data', default= '../data/val_image',
                        help='provide absolute path to validation data')
    parser.add_argument('--val_label', default= '../data/coordinates/translated_coords_val.csv',
                        help='provide relative path to validation labels')
    parser.add_argument('--batch_size', default=10, help='provide size of batch')
    parser.add_argument('--epochs', default=100, help = 'provide number of epochs to train')
    parser.add_argument('--ckpt', default='../model_ckpts', help = 'provide path to save model checkpoints')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':

    main()
