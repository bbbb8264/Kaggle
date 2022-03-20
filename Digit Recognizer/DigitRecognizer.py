import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from PIL import Image

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
train = tf.random.shuffle(df_train)
x_train = tf.reshape(train[:, 1:], (len(train), 28, 28, 1))
x_train = x_train/255
x_val = x_train[40000:]
x_train = x_train[:40000]
x_test = tf.reshape(df_test, (len(df_test), 28, 28, 1))
x_test = x_test/255
y_train = tf.one_hot(train[:, 0], 10)
y_val = y_train[40000:]
y_train = y_train[:40000]
model = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation='softmax'),
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('loss') < 0.006:
            print("\n reached loss=0.006, hence stopped training!!")
            self.model.stop_training = True

callbacks = MyCallback()
model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), callbacks=[callbacks])
output = tf.argmax(model.predict(x_test), axis=1)
output = {"ImageId": list(range(1, 28001)), "Label": output}
output = pd.DataFrame(output)
output.to_csv("submit.csv", index=False)
