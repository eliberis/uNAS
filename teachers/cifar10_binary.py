import tensorflow as tf
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from dataset import CIFAR10


def mobilenet_v2_like(input_shape, num_classes):
    def _inverted_res_block(i, filters, alpha, stride, expansion, block_id):
        prefix = 'block_{}_'.format(block_id)
        in_channels = i.shape[-1]
        x = i

        # Expand
        x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='valid',
                          use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(name=prefix + 'expand_BN')(x)  # epsilon=1e-3, momentum=0.999,
        x = layers.ReLU(name=prefix + 'expand_relu')(x)

        # Depthwise
        x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                                   use_bias=False, padding='same', name=prefix + 'depthwise')(x)
        x = layers.BatchNormalization(name=prefix + 'depthwise_BN')(x)  # epsilon=1e-3, momentum=0.999,
        x = layers.ReLU(name=prefix + 'depthwise_relu')(x)

        # Project
        pointwise_filters = int(filters * alpha)
        x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='valid', use_bias=False,
                          activation=None, name=prefix + 'project')(x)
        x = layers.BatchNormalization(name=prefix + 'project_BN')(x)  # epsilon=1e-3, momentum=0.999,

        if stride == 1:
            if in_channels != pointwise_filters:
                i = layers.Conv2D(pointwise_filters, kernel_size=1, padding='valid', use_bias=False,
                                  activation=None, name=prefix + 'adjust')(i)
            x = layers.Add(name=prefix + 'add')([i, x])
        return x

    i = layers.Input(shape=input_shape)
    x = i

    alpha = 1.0
    x = layers.Conv2D(32, kernel_size=3, strides=(1, 1),
                      padding='same', use_bias=False, name='Conv1')(x)
    x = layers.BatchNormalization(name='bn_Conv1')(x)  # epsilon=1e-3, momentum=0.999,
    x = layers.ReLU(name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)
    # TODO: Dropout?

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)
    # TODO: Dropout?

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    # TODO: Dropout?

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)
    # TODO: Dropout?

    x = layers.Conv2D(1280, kernel_size=1, use_bias=False, name='Conv_1', padding="valid")(x)
    x = layers.BatchNormalization(name='Conv_1_bn')(x)  # epsilon=1e-3, momentum=0.999,
    x = layers.ReLU(name='out_relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, name='predictions')(x)

    return Model(i, x)

# 0.9213 Test accuracy


def main():
    dataset = CIFAR10(binary=True, validation_split=0.0)  # not using validation for anything
    model = mobilenet_v2_like(dataset.input_shape, dataset.num_classes)

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=SGDW(lr=0.01, momentum=0.9, weight_decay=1e-5),
                  metrics=['accuracy'])
    model.summary()

    batch_size = 128

    train_data = dataset.train_dataset() \
        .shuffle(8 * batch_size) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    valid_data = dataset.test_dataset() \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def lr_schedule(epoch):
        if 0 <= epoch < 35:
            return 0.01
        if 35 <= epoch < 65:
            return 0.005
        return 0.001

    model.fit(train_data, validation_data=valid_data, epochs=80,
              callbacks=[LearningRateScheduler(lr_schedule)])
    model.save("cnn-cifar10-binary.h5")


if __name__ == '__main__':
    main()
