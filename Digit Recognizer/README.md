Dataset: https://www.kaggle.com/c/digit-recognizer/data

Conduct with simple CNN
tf.keras.layers.RandomRotation(0.2),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tf.keras.layers.MaxPool2D((2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation="relu"),
tf.keras.layers.Dense(64, activation="relu"),
tf.keras.layers.Dense(10, activation='softmax')

Score: 0.98257