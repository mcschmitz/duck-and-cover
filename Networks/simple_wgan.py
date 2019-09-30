import functools

from keras import Model
from keras.layers import *
from keras.optimizers import Adam

from Networks import DaCSimple
from Networks import wasserstein_loss, RandomWeightedAverage, gradient_penalty_loss


class DaCSimpleWGan(DaCSimple):

    def __init__(self, img_height: int, img_width: int, channels: int = 3, latent_size: int = 128,
                 batch_size: int = 64):
        """
        @TODO
        Args:
            img_height:
            img_width:
            channels:
            latent_size:
        """
        super(DaCSimpleWGan, self).__init__(img_height, img_width, channels, latent_size)
        self.discriminator_model = None
        self.discriminator_loss = []
        self.batch_size = batch_size

    def build_models(self, gradient_penalty_weight, optimizer=Adam(beta_1=0, beta_2=0.99)):
        #  TODO Refactor
        """
        @TODO
        Args:
            gradient_penalty_weight:
            optimizer:

        Returns:

        """
        self.discriminator = self.build_discriminator()

        noise_input = Input((self.latent_size,), name="Noise_Input")
        image_input = Input(self.img_shape, name="Img_Input")

        self.generator = self.build_generator()

        generated_image = self.generator(noise_input)
        averaged_samples = RandomWeightedAverage(self.batch_size)([image_input, generated_image])
        discriminated_real = self.discriminator(image_input)
        discriminated_fake = self.discriminator(generated_image)
        discriminated_avg = self.discriminator(averaged_samples)
        partial_gp_loss = functools.partial(gradient_penalty_loss, averaged_samples=averaged_samples,
                                            gradient_penalty_weight=gradient_penalty_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model = Model(inputs=[image_input, noise_input],
                                         outputs=[discriminated_real, discriminated_fake, discriminated_avg])
        self.discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                                         optimizer=optimizer)

        self.discriminator.trainable = False

        eval_of_gen_image = self.discriminator(generated_image)
        self.adversarial_model = Model(noise_input, eval_of_gen_image)
        self.adversarial_model.compile(loss=[wasserstein_loss], optimizer=optimizer)
        self.adversarial_model.n_epochs = 0

    def build_discriminator(self):
        """
        @TODO
        Returns:

        """
        image_input = Input(self.img_shape)
        x = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer=initializers.zero(),
                   kernel_initializer=initializers.random_normal(stddev=1))(image_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', bias_initializer=initializers.zero(),
                   kernel_initializer=initializers.random_normal(stddev=1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        cur_img_size = self.img_shape[0] // 2
        n_kernels = 64
        while cur_img_size > 4:
            x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', bias_initializer=initializers.zero(),
                       kernel_initializer=initializers.random_normal(stddev=1))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)
            n_kernels *= 2
            cur_img_size //= 2

        x = Flatten()(x)

        discriminator_output = Dense(1)(x)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def train_on_batch(self, real_images, ratio: int = 5):
        """
        @TODO
        Args:
            real_images:
            ratio:

        Returns:

        """
        fake = np.ones((len(real_images), 1)) * -1
        real = np.ones((len(real_images), 1))
        dummy_y = np.zeros((len(real_images), 1), dtype=np.float32)

        noise = np.random.normal(size=(len(real_images), self.latent_size))

        self.generator_loss.append([self.adversarial_model.train_on_batch(noise, real)])

        for _ in range(ratio):
            losses = self.discriminator_model.train_on_batch([real_images, noise], [real, fake, dummy_y])
        self.discriminator_loss.append(losses)

        return np.array(self.discriminator_loss).mean(axis=0), np.mean(self.generator_loss)

    def __reset_discriminator_loss(self):
        """
        @TODO
        Returns:

        """
        self.discriminator_loss = []

    def reset_metrics(self):
        """
        @TODO
        Returns:

        """
        self.__reset_generator_loss()
        self.__reset_discriminator_accuracy()
        self.__reset_discriminator_loss()
