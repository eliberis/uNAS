import logging
from typing import Optional

import tensorflow as tf

from config import TrainingConfig
from pruning import DPFPruning
from utils import debug_mode


class ModelTrainer:
    """Trains Keras models according to the specified config."""
    def __init__(self, training_config: TrainingConfig):
        self.log = logging.getLogger("Model trainer")
        self.config = training_config
        self.distillation = training_config.distillation
        self.pruning = training_config.pruning
        self.dataset = training_config.dataset

    def train_and_eval(self, model: tf.keras.Model,
                       epochs: Optional[int] = None, sparsity: Optional[float] = None):
        """
        Trains a Keras model and returns its validation set error (1.0 - accuracy).
        :param model: A Keras model.
        :param epochs: Overrides the duration of training.
        :param sparsity: Desired sparsity level (for unstructured sparsity)
        :returns Smallest error on validation set seen during training, the error on the test set,
        pruned weights (if pruning was used)
        """
        dataset = self.config.dataset
        batch_size = self.config.batch_size
        sparsity = sparsity or 0.0

        train = dataset.train_dataset() \
            .shuffle(batch_size * 8) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        val = dataset.validation_dataset() \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # TODO: check if this works, make sure we're excluding the last layer from the student
        if self.pruning and self.distillation:
            raise NotImplementedError()

        if self.distillation:
            teacher = tf.keras.models.load_model(self.distillation.distill_from)
            teacher._name = "teacher_"
            teacher.trainable = False

            t, a = self.distillation.temperature, self.distillation.alpha

            # Assemble a parallel model with the teacher and student
            i = tf.keras.Input(shape=dataset.input_shape)
            cxent = tf.keras.losses.CategoricalCrossentropy()

            stud_logits = model(i)
            tchr_logits = teacher(i)

            o_stud = tf.keras.layers.Softmax()(stud_logits / t)
            o_tchr = tf.keras.layers.Softmax()(tchr_logits / t)
            teaching_loss = (a * t * t) * cxent(o_tchr, o_stud)

            model = tf.keras.Model(inputs=i, outputs=stud_logits)
            model.add_loss(teaching_loss, inputs=True)

        if self.dataset.num_classes == 2:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        model.compile(optimizer=self.config.optimizer(),
                      loss=loss, metrics=[accuracy])

        # TODO: adjust metrics by class weight?
        class_weight = {k: v for k, v in enumerate(self.dataset.class_weight())} \
            if self.config.use_class_weight else None
        epochs = epochs or self.config.epochs
        callbacks = self.config.callbacks()
        check_logs_from_epoch = 0

        pruning_cb = None
        if self.pruning and sparsity > 0.0:
            assert 0.0 < sparsity <= 1.0
            self.log.info(f"Target sparsity: {sparsity:.4f}")
            pruning_cb = DPFPruning(target_sparsity=sparsity, structured=self.pruning.structured,
                                    start_pruning_at_epoch=self.pruning.start_pruning_at_epoch,
                                    finish_pruning_by_epoch=self.pruning.finish_pruning_by_epoch)
            check_logs_from_epoch = self.pruning.finish_pruning_by_epoch
            callbacks.append(pruning_cb)

        log = model.fit(train, epochs=epochs, validation_data=val,
                        verbose=1 if debug_mode() else 2,
                        callbacks=callbacks, class_weight=class_weight)

        test = dataset.test_dataset() \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        _, test_acc = model.evaluate(test, verbose=0)

        return {
            "val_error": 1.0 - max(log.history["val_accuracy"][check_logs_from_epoch:]),
            "test_error": 1.0 - test_acc,
            "pruned_weights": pruning_cb.weights if pruning_cb else None
        }
