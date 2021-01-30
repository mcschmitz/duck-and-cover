import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter


class GradientAccumulatorModel(Model):
    def __init__(self, gradient_accumulation_steps: int = 1, *args, **kwargs):
        super(GradientAccumulatorModel, self).__init__(*args, **kwargs)
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        use_weights = sample_weight is not None
        x = x if isinstance(x, tuple) else [x]
        y = y if isinstance(y, tuple) else [y]
        sample_weight = (
            sample_weight if isinstance(x, tuple) else [sample_weight]
        )

        batch_size = x[0].shape[0]
        minibatch_size = batch_size // self.gradient_accumulation_steps

        train_vars = self.trainable_variables
        accum_gradient = [tf.zeros_like(v) for v in train_vars]
        for step in range(self.gradient_accumulation_steps):
            start = step * minibatch_size
            end = start + minibatch_size
            x_step = [xi[start:end] for xi in x]
            y_step = [yi[start:end] for yi in y]
            weights_step = (
                [weightsi[start:end] for weightsi in sample_weight]
                if use_weights
                else None
            )
            with backprop.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(train_vars)
                y_pred = self(x_step, training=True)
                loss = self.compiled_loss(
                    y_step,
                    y_pred,
                    weights_step,
                    regularization_losses=self.losses,
                )
            gradients = tape.gradient(loss, train_vars)
            accum_gradient = [
                (acum_grad + grad)
                for acum_grad, grad in zip(accum_gradient, gradients)
            ]
        accum_gradient = [
            grad / self.gradient_accumulation_steps for grad in accum_gradient
        ]
        self.optimizer.apply_gradients(zip(accum_gradient, train_vars))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


class GradientAccumulatorSequential(Sequential):
    def __init__(
        self, gradient_accumulation_steps: int, layers=None, name=None
    ):
        super(GradientAccumulatorSequential, self).__init__(
            layers=layers, name=name
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        use_weights = sample_weight is not None
        x = x if isinstance(x, tuple) else [x]
        y = y if isinstance(y, tuple) else [y]
        sample_weight = (
            sample_weight if isinstance(x, tuple) else [sample_weight]
        )

        batch_size = x[0].shape[0]
        minibatch_size = batch_size // self.gradient_accumulation_steps

        train_vars = self.trainable_variables
        accum_gradient = [tf.zeros_like(v) for v in train_vars]
        for step in range(self.gradient_accumulation_steps):
            start = step * minibatch_size
            end = start + minibatch_size
            x_step = [xi[start:end] for xi in x]
            y_step = [yi[start:end] for yi in y]
            weights_step = (
                [weightsi[start:end] for weightsi in sample_weight]
                if use_weights
                else None
            )
            with backprop.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(train_vars)
                y_pred = self(x_step, training=True)
                loss = self.compiled_loss(
                    y_step,
                    y_pred,
                    weights_step,
                    regularization_losses=self.losses,
                )
            gradients = tape.gradient(loss, train_vars)
            accum_gradient = [
                (acum_grad + grad)
                for acum_grad, grad in zip(accum_gradient, gradients)
            ]
        accum_gradient = [
            grad / self.gradient_accumulation_steps for grad in accum_gradient
        ]
        self.optimizer.apply_gradients(zip(accum_gradient, train_vars))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
