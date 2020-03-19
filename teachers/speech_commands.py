import tensorflow as tf
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, ReLU, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model

from dataset import SpeechCommands

# Acc: 0.9542 on the test set


def get_model(input_shape, num_classes):
    def conv_bn_relu(filters, kernel_size, strides=(1, 1), padding="valid"):
        def apply(x):
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
        return apply

    i = Input(shape=input_shape)
    x = i

    x = conv_bn_relu(276, kernel_size=(10, 4), strides=(2, 1), padding="same")(x)
    x = conv_bn_relu(276, kernel_size=3, strides=(2, 2))(x)
    x = conv_bn_relu(276, kernel_size=3, strides=(1, 1))(x)
    x = conv_bn_relu(276, kernel_size=3, strides=(1, 1))(x)
    x = conv_bn_relu(276, kernel_size=3, strides=(1, 1))(x)
    x = conv_bn_relu(276, kernel_size=3, strides=(1, 1))(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(num_classes)(x)

    return Model(i, x)


def main():
    dataset = SpeechCommands("/datasets/speech_commands_v0.02")
    model = get_model(dataset.input_shape, dataset.num_classes)

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=AdamW(lr=0.0005, weight_decay=1e-5),
                  metrics=['accuracy'])
    model.summary()

    batch_size = 100

    train_data = dataset.train_dataset() \
        .shuffle(8 * batch_size) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    valid_data = dataset.test_dataset() \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def lr_schedule(epoch):
        if 0 <= epoch < 20:
            return 0.0005
        if 20 <= epoch < 40:
            return 0.0001
        return 0.00002

    model.fit(train_data, validation_data=valid_data, epochs=50, callbacks=[
        LearningRateScheduler(lr_schedule)
    ])
    model.save("cnn-speech-commands.h5")


if __name__ == '__main__':
    main()
