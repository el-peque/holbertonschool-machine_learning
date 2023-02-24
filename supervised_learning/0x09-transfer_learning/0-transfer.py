#!/usr/bin/env python3
"""Transfer Knowledge"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """Pre-processes the data for inception v3 inspired model"""
    X_p = K.applications.inception_v3.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    X_p_train, Y_p_train = preprocess_data(x_train, y_train)
    test_ds = preprocess_data(x_test, y_test)
    
    inception = K.applications.InceptionV3(include_top=False,
                                           weights="imagenet",
                                           input_shape=(224, 224, 3))
    inputs = K.Input(shape=(32, 32, 3))
    X = K.layers.Resizing(224, 224)(inputs)
    X = inception(X, training=False)
    X = K.layers.Dense(100, activation='relu')(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Dropout(0.2)(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)
    inception.trainable = False
    
    model = K.Model(inputs, outputs)
    model.summary()
    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_p_train, Y_p_train, epochs=3, batch_size=100, validation_data=test_ds)
    model.save('cifar10.h5')

