import tensorflow as tf

from tensorflow.python.keras import layers, models

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import scipy


data_dir = '../Visualization/images_original'


# Assuming 'data_dir' is defined and points to the directory with your images

data_generator = ImageDataGenerator(rescale=1./255)
data = data_generator.flow_from_directory(
    directory=data_dir,
    batch_size=20,           # Number of images to load at each iteration
    class_mode='categorical',# For multi-class classification
    shuffle=True)            

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']

# Fetch a single batch of images
images, labels = next(data)


def plotShit():
    plt.figure()
    for i in range(25):
        if i >= len(images):  # Check to ensure there are enough images
            break
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        # Find the index of the one in label's one-hot encoding
        label_index = int(labels[i].argmax())
        plt.xlabel(class_names[label_index])
    plt.show()



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(432, 288, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(data.image_data_generator, epochs=10, 
                    validation_data=data.labels)


