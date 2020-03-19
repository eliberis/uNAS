import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

from dataset import Chars74K


def get_model(input_shape, num_classes):
    def conv_bn_relu(filters, kernel_size, stride=1):
        def apply(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=(stride, stride))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
        return apply

    i = Input(shape=input_shape)
    x = i

    x = conv_bn_relu(64, kernel_size=5, stride=1)(x)

    x = conv_bn_relu(96, kernel_size=3, stride=2)(x)
    x = conv_bn_relu(96, kernel_size=3, stride=1)(x)
    x = conv_bn_relu(96, kernel_size=3, stride=1)(x)

    x = conv_bn_relu(192, kernel_size=3, stride=2)(x)
    x = conv_bn_relu(192, kernel_size=3, stride=1)(x)
    x = conv_bn_relu(192, kernel_size=3, stride=1)(x)

    x = conv_bn_relu(256, kernel_size=3, stride=1)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.65)(x)
    x = Dense(1024, activation="relu")(x)

    x = Dropout(0.65)(x)
    x = Dense(512, activation="relu")(x)

    x = Dropout(0.65)(x)
    x = Dense(512, activation="relu")(x)

    x = Dense(num_classes)(x)

    return Model(i, x)


def main():
    dataset = Chars74K("/datasets/chars74k", img_size=(48, 48), validation_split=0.0)  # not using validation for anything
    model = get_model(dataset.input_shape, dataset.num_classes)

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=AdamW(lr=0.001, weight_decay=5e-4),
                  metrics=['accuracy'])
    model.summary()

    batch_size = 128

    train_data = dataset.train_dataset() \
        .shuffle(8 * batch_size) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    valid_data = dataset.test_dataset() \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model.fit(train_data, validation_data=valid_data, epochs=200)
    model.save("cnn-chars74k.h5")


if __name__ == '__main__':
    main()
