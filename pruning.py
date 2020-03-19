import tensorflow as tf
import numpy as np
import logging


@tf.custom_gradient
def _norm_axis_0(x):
    y = tf.norm(x, axis=0)

    def grad(dy):
        return dy * (x / (y + 1e-19))

    return y, grad


class DPFPruning(tf.keras.callbacks.Callback):
    """ An implementation of Dynamic Model Pruning with Feedback by Lin et al.
        This is an unstructured pruning method (produces sparse weight matrices).
        https://openreview.net/forum?id=SJem8lSFwB
    """

    def __init__(self, weight_predicate=None,
                 structured=False,
                 target_sparsity=0.90,
                 start_pruning_at_epoch=0,
                 finish_pruning_by_epoch=None,
                 update_iterations=16,
                 diagnostics=False):
        """
        Creates a Keras callback that prunes the model by computing gradients on the sparse
        weights (weights multiplied by a pruning matrix), but applying them to dense weights.
        Masks are determined by magnitude pruning, so using an L1/L2 regularisation or weight
        decay is recommended.
        Pruning starts at sparsity level of 0 and gradually increases every epoch.
        Note: due to using both dense and sparse weights, 2 sets of weights will be stored,
        which will increase the memory usage.
        :param weight_predicate: A predicate (tf.Variable x keras Model to boolean function) to
        determine  whether a weight matrix should be pruned. A collection of prunable weights will
        be determined at the start of training by calling this predicate on every variable in
        `model.trainable_weights`. A good default behaviour is to include only kernel matrices
        (don't prune biases) and not prune the last layer of the model.
        :param structured: Whether to produce a sparse model (`structured = False`, default) or
        prune the model channel-wise (`structured = True`).
        :param target_sparsity: A level of sparsity to reach by the end of training, computed
        as the fraction of values (or channels, if `structured = True`) set to 0.
        :param start_pruning_at_epoch: An epoch at which to start pruning.
        :param finish_pruning_by_epoch: An epoch at which to finish pruning.
        :param update_iterations: After how many batches pruning matrices should be updated.
        :param diagnostics: Whether to print diagnostic logs
        """
        assert 0.0 < target_sparsity < 1.0
        assert start_pruning_at_epoch <= finish_pruning_by_epoch
        super().__init__()

        # Internally, we count epochs from 0
        self.t_0 = start_pruning_at_epoch - 1
        self.t_n = finish_pruning_by_epoch - 1
        self.p = update_iterations
        self.e = 3  # Exponent in the sparsity schedule expression

        self.filter = weight_predicate
        if weight_predicate is None:
            self.filter = \
                self._default_structured_predicate if structured \
                else self._default_unstructured_predicate
        self.update_masks = \
            self._compute_structured_pruning_masks if structured \
            else self._compute_unstructured_pruning_masks
        self.iterations = 0
        self.epoch = 0
        self.s_f = target_sparsity
        self.s_i = 0
        self.s_t = self.s_i

        self.weights = []
        self.masks = []
        self.dense_weight_values = []

        self.log = logging.getLogger("DPFPruner")
        self.diagnostics = diagnostics

    def _default_unstructured_predicate(self, w, model):
        return "kernel" in w.name \
               and (w.name not in [m.name for m in model.layers[-1].weights])

    def _default_structured_predicate(self, w, model):
        return self._default_unstructured_predicate(w, model) and w.shape[-1] > 1 \
               and (w.name not in [m.name for c in model.layers[0].outbound_nodes
                                   for m in c.outbound_layer.weights])

    def _should_prune_this_epoch(self):
        return self.t_0 <= self.epoch <= self.t_n

    def on_train_begin(self, logs=None):
        # The callback mechanism doesn't allow for pre-optimiser callbacks to modify gradients.
        # Instead, the intervention is accomplished by monkey-patching the `apply_gradients` method
        # on `model.optimizer`. (Would have been nicer to create a transparent proxy for the
        # optimizer and use that in model.compile, but this approach would do for now).

        # Store the original `apply_gradients` function in an unlikely-to-name-clash field
        # and patch-in our substitute
        self.model.optimizer.original_apply_gradients__ = self.model.optimizer.apply_gradients

        def apply_gradients(optimizer, grads_and_vars, **kwargs):
            self.on_before_gradient_application()
            return optimizer.original_apply_gradients__(grads_and_vars, **kwargs)

        self.model.optimizer.apply_gradients = \
            apply_gradients.__get__(self.model.optimizer, tf.keras.optimizers.Optimizer)

        # Determine the number of epochs to fully prune the network by
        if self.t_n is None:
            self.t_n = self.params["epochs"]

        # Keep a reference to weight variables
        self.weights = [w for w in self.model.trainable_weights if self.filter(w, self.model)]

        # The following could just be tensors,
        # but we use variables to avoid multiple alloc / dealloc
        self.masks = [tf.Variable(tf.ones_like(w), trainable=False) for w in self.weights]  # sparsity masks
        self.dense_weight_values = [tf.Variable(w, trainable=False) for w in self.weights]  # cached dense weights

    def _cache_weights(self):
        for w, d in zip(self.weights, self.dense_weight_values):
            d.assign(w)

    def _apply_masks(self):
        for w, m in zip(self.weights, self.masks):
            w.assign(w * m)

    def _restore_weights(self):
        for w, d in zip(self.weights, self.dense_weight_values):
            w.assign(d)

    def on_train_end(self, logs=None):
        # Undo the monkey-patch
        self.model.optimizer.apply_gradients = self.model.optimizer.original_apply_gradients__

        # Return only sparse weights at the end
        self._apply_masks()

    def on_test_begin(self, logs=None):
        self._cache_weights()
        self._apply_masks()

    def on_test_end(self, logs=None):
        # self._restore_weights()
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.iterations = 0

        if self._should_prune_this_epoch():
            n = self.t_n - self.t_0
            self.s_t = self.s_f + (self.s_i - self.s_f) * pow(1.0 - (self.epoch - self.t_0) / n, self.e)

    def on_epoch_end(self, epoch, logs=None):
        self._print_diagnostics()

    def _print_diagnostics(self):
        if not self.diagnostics:
            return

        self.log.warning(f"At the end of epoch #{self.epoch + 1}:")
        total_kept, total_size = 0, 0
        for w, m in zip(self.weights, self.masks):
            kept = tf.math.count_nonzero(m)
            size = np.prod(m.shape)

            sparsity = (size - kept) / size
            self.log.warning(f"Matrix {w.name}: {kept}/{size} " +
                             (f"({sparsity:.4f} pruned)" if kept > 0 else "(fully pruned!)"))
            total_kept += kept
            total_size += size
        total_sparsity = (total_size - total_kept) / total_size
        self.log.warning(f"Total: {total_kept}/{total_size} ({total_sparsity:.4f} pruned)")

    def _compute_unstructured_pruning_masks(self):
        """ Computes unstructed pruning masks, where `self.s_t` proportion of values
            are set to 0 for specified matrices. """
        # Compute weight saliences
        # Magnitude pruning (unfortunately): weight saliency is its abs
        saliences = [tf.abs(w) for w in self.weights]

        # Pick a global threshold that removes `self.s_t` fraction of weights
        all_sals = tf.concat([tf.reshape(s, (-1,)) for s in saliences], axis=0)
        threshold = np.quantile(all_sals, self.s_t)

        for m, s in zip(self.masks, saliences):
            m.assign(tf.cast(s >= threshold, tf.float32))

    def _compute_structured_pruning_masks(self):
        """ Computes structured pruning masks, where where `self.s_t` proportion of _channels_ (or units)
            are pruned away by setting the mask values for an entire channel to 0. """
        # This is a norm-based filter pruning algorithm: compute the L2 norm of each channel
        # This assumes that self.weights _only_ contains Conv2D 4-dimensional (K, K, C_in, C_out)
        # matrices and Dense/FullyConnected (U_in, U_out) matrices
        saliences = [tf.norm(tf.reshape(w, (-1, w.shape[-1])), ord=2, axis=0) for w in self.weights]

        all_sals = tf.concat(saliences, axis=0)

        # The following could work, too, if we need a pure TF solution
        # from tensorflow.python.ops.nn_ops import nth_element
        # n = tf.cast(tf.math.round(self.s_t * all_sals.shape[0]), tf.int32)
        # threshold = nth_element(all_sals, n)

        threshold = np.quantile(all_sals, self.s_t)

        for m, s in zip(self.masks, saliences):
            m.assign(tf.ones_like(m) * tf.cast(s >= threshold, tf.float32))

    def on_train_batch_begin(self, batch, logs=None):
        if self.iterations % self.p == 0 and self._should_prune_this_epoch():
            self.update_masks()

        # Gradient should be computed with sparse variables --- so save dense values
        self._cache_weights()

        # Now sparsify the weights before continuing with the batch
        self._apply_masks()

    def on_before_gradient_application(self):
        # Restore dense weights
        self._restore_weights()

    def on_train_batch_end(self, batch, logs=None):
        self.iterations += 1
        logs["sparsity"] = self.s_t
