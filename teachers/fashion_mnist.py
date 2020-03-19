import tensorflow as tf
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, ReLU, \
    GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

from dataset import FashionMNIST


def get_model(input_shape, num_classes):
    def conv_bn_relu(filters, kernel_size, stride=1, padding="valid"):
        def apply(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=(stride, stride),
                       padding=padding)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
        return apply

    i = Input(shape=input_shape)
    x = i

    x = conv_bn_relu(48, kernel_size=3, stride=1, padding="same")(x)
    x = conv_bn_relu(48, kernel_size=3, stride=1, padding="same")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = conv_bn_relu(96, kernel_size=3, stride=1, padding="same")(x)
    x = conv_bn_relu(96, kernel_size=3, stride=1, padding="same")(x)
    x = conv_bn_relu(96, kernel_size=3, stride=1, padding="same")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.35)(x)

    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes)(x)

    return Model(i, x)


def main():
    dataset = FashionMNIST()
    model = get_model(dataset.input_shape, dataset.num_classes)

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=AdamW(lr=0.001, weight_decay=5e-5),
                  metrics=['accuracy'])
    model.summary()

    batch_size = 128

    train_data = dataset.train_dataset() \
        .shuffle(8 * batch_size) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    valid_data = dataset.test_dataset() \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model.fit(train_data, validation_data=valid_data, epochs=40, callbacks=[
        LearningRateScheduler(lambda e: 0.001 if e < 25 else 0.0001)
    ])
    model.save("cnn-fashion-mnist.h5")


if __name__ == '__main__':
    main()
