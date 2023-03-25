import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import logging

# Define o formato da mensagem de log
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=log_format)
# Configura o nível de log e o formato da mensagem
logger = logging.getLogger("Training_CIFAR10_CNN_model")
formatter = logging.Formatter(log_format)
fh = logging.FileHandler('cifar.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("Loading the CIFAR-10 dataset")
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)

logger.info("Preprocessing the data")
# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

logger.info("Defining the model architecture")
# Define the CNN model architecture
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

logger.info("Compiling the model")
# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

logger.info("Training the model")
# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.1)

logger.info("Evaluating the model")
# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

logger.info("Saving the model")
model.save("cifar10_model.h5")
