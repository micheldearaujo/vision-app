import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('models/cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
import logging
import datetime

# Define o formato da mensagem de log
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configura o nível de log e o formato da mensagem
logging.basicConfig(level=logging.DEBUG, format=log_format)


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to display the classification result and the most important features
def display_result(image, class_name):

    conv_layer_model = tf.keras.models.Model(
    model.inputs,
    model.layers[0].output
    )
    conv_output = conv_layer_model(image)
    conv_output = np.squeeze(conv_output)

    # Plot the output of the first convolutional layer

    fig, axs = plt.subplots(4, 8, figsize=(15, 12))

    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(conv_output[:, :, i], cmap='gray')
        plt.axis('off')

    # Display the most important features
    st.write('#### Results of the first Convolutional Filter')
    st.pyplot(fig)

# Streamlit app
def app():
    st.title('CIFAR-10 Image Classification')

    # File upload
    file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if file is not None:

        # Preprocess the image
        img_array = preprocess_image(file)

        # Make predictions
        predictions = model.predict(img_array)
        logging.debug("The predictied probabilities are: \n{}".format(list(predictions)))
        # Get the index of the highest probability
        class_index = np.argmax(predictions)
        logging.debug("The predicted class index is: {}".format(class_index))
        # Get the class name by using the index
        class_name = class_names[class_index]
        logging.debug("The predicted class name is: {}".format(class_names[class_index]))

        # Display the original Image
        st.image(file, caption=f'Classified as {class_name}', use_column_width=True)
        # Display the result
        display_result(img_array, class_name)

if __name__ == '__main__':
    app()
