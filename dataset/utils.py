import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


def grid_visualise_25(dataset: tf.data.Dataset):
    plt.figure(figsize=(10, 10))

    for i, (img, label) in enumerate(dataset.take(25)):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if img.shape[-1] == 1:
            plt.imshow(img[:, :, 0], cmap=plt.cm.binary)
        else:
            plt.imshow(img)
        plt.xlabel(f"Class {label}")
    plt.show()


def with_probability(p, true, false):
    return tf.cond(tf.random.uniform([]) < p, true, false)


def random_rotate(x, rads=0.3):
    angle = tf.random.uniform([], minval=-rads, maxval=rads)
    return tfa.image.rotate(x, angle)


def random_shift(x, h_pixels, w_pixels):
    orig = x.shape
    x = tf.pad(x, mode="SYMMETRIC",
               paddings=tf.constant([[w_pixels, w_pixels], [h_pixels, h_pixels], [0, 0]]))
    return tf.image.random_crop(x, size=orig)
