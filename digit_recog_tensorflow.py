import pandas as pd
import numpy as np
np.random.seed(1337)
from matplotlib import pyplot as plt
import tensorflow as tf

def load_split_data(self):
    data = pd.read_csv('digit/train.csv')
    test = pd.read_csv('digit/test.csv')
    data_label = data['label']
    x_train, x_test, x_train_label, x_test_label = tf.keras.utils.train_test_split(data, data_label, test_size=0.20, random_state=42)
    x_train = x_train.drop(columns=['label'])
    x_test = x_test.drop(columns=['label'])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_test,x_test_label,x_train,x_train_label

def load_from_mnist(self):
    mnist=tf.keras.datasets.mnist
    (x_train, x_train_label),(x_test, x_test_label) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_test,x_test_label,x_train,x_train_label


def gen_model(self):
    self.model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding="same", input_shape=(28, 28, 1),
                                  data_format="channels_last", activation="relu"))
    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding="valid", activation="relu"))
    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    self.model.add(tf.keras.layers.Flatten())
    self.model.add(tf.keras.layers.Dense(120, input_shape=(400,), activation='relu'))
    self.model.add(tf.keras.layers.Dense(84, input_shape=(120,), activation='relu'))
    self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

def train_model(self):
    self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#    self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])load_from_mnist
    self.model.fit(self.x_train, self.x_train_label, epochs=self.no_of_epochs, batch_size=self.batch_size)

def save_model(self):
    model_json = self.model.to_json()
    with open("%s.json" % self.save_model_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    self.model.save_weights("%s.h5" % self.save_model_name)

def predict_output(self):
    # loss_and_metrics = model.evaluate(x_test, x_test_label,batch_size=32)
    classes = self.model.predict(self.x_test, batch_size=self.batch_size)
    classes = classes.argmax(axis=1)
    for row in range(0, 3):
        plt.title("label=%s" % classes[row])
        plt.imshow(np.reshape(self.x_test[row], (28, 28)), cmap='gray')
        plt.show()

def encode_data(self):
    self.x_train_label = tf.keras.utils.to_categorical(self.x_train_label, self.no_of_classes)
    self.x_test_label = tf.keras.utils.to_categorical(self.x_test_label, self.no_of_classes)

def load_model(self):
    print("Loading model from disk")
    # load json and create model
    json_file = open('%s.json' % self.save_model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    self.model.load_weights("%s.h5" % self.save_model_name)
    print("Loaded model from disk")

class digit_recognition:
    def __init__(self):
        self.model=tf.keras.models.Sequential()
#        self.x_test,self.x_test_label,self.x_train,self.x_train_label=self.load_split_data()
        self.x_test, self.x_test_label, self.x_train, self.x_train_label = self.load_from_mnist() #load data from mnist
        self.no_of_epochs=5
        self.batch_size=300
        self.save_model_name='lenet'
        self.no_of_classes=10
    load_split_data=load_split_data
    encode_data=encode_data
    gen_model=gen_model
    train_model=train_model
    predict_output=predict_output
    save_model=save_model
    load_model=load_model
    load_from_mnist=load_from_mnist


def digit_recog_crt():
    digit=digit_recognition()
    digit.encode_data()
    digit.gen_model()
    digit.train_model()
    digit.predict_output()
    digit.save_model()

def digit_recog_reuse():
    digit = digit_recognition()
    digit.encode_data()
    digit.load_model()
    digit.predict_output()

if __name__ == '__main__':
    digit_recog_crt()
    digit_recog_reuse()
