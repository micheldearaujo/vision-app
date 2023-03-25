import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from kerastuner.tuners import RandomSearch
import time

def build_model(hp):
    model = keras.Sequential()

    # Tune the number of filters in the first convolutional layer
    hp_filters = hp.Int('filters', min_value=32, max_value=256, step=32)
    model.add(layers.Conv2D(hp_filters, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    # Tune the number of convolutional blocks and the number of filters in each block
    for i in range(hp.Int('num_blocks', min_value=1, max_value=3)):
        hp_block_filters = hp.Int('block_filters_' + str(i), min_value=32, max_value=256, step=32)
        model.add(layers.Conv2D(hp_block_filters, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    # Tune the number of units in the fully connected layer
    hp_units = hp.Int('units', min_value=128, max_value=512, step=64)
    model.add(layers.Dense(hp_units, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))

    # Tune the learning rate and the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])

    model.compile(optimizer=hp_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model():
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    # Define the hyperparameter search space
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        directory='hyperparameters',
        project_name='cifar10')

    # Start the hyperparameter search
    start_time = time.time()
    tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    elapsed_time = time.time() - start_time

    # Print the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    # Build and train the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    return model

if __name__ == '__main__':
    model = train_model()
    model.save('models/cifar10_model_tunned.h5')