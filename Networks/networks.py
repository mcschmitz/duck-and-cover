import functools

from keras import Model
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, Adadelta

from Networks.utils import wasserstein_loss, RandomWeightedAverage, gradient_penalty_loss, PixelNorm


#  TODO Use Wasserstein Loss
#  TODO Add Release Year information
#  TODO Add Genre Information
#  TODO Let GAN grow

class CoverGAN:

    def __init__(self, img_height: int, img_width: int, channels: int = 3, latent_size: int = 128):
        """
        Simple GAN that builds a discriminator a generator and the adversarial model to train the GAN based on the
        binary crossentropy loss for the generator and the discriminator

        Args:
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the image
        """
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_size = latent_size
        self.discriminator = None
        self.generator = None
        self.discriminator_model = None
        self.combined_model = None
        self.generator_loss = []
        self.discriminator_accuracy = []

    def build_models(self, optimizer=Adam(beta_1=0, beta_2=0.99)):
        """
        @TODO
        Builds the desired GAN that allows to generate covers.

        The GAN can either be a simple GAN or a WGAN using Wasserstein loss with gradient penalty to improve
        learning. Also additional information can be passed to the GAN like release year information.

        Args:
            optimizer: Which optimizer to use
            simple GAN with binary crossentropy loss
        """
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        self.discriminator.trainable = False

        self._build_combined_model(optimizer)

        self.discriminator.trainable = True
        self.generator.trainable = False

        self._build_discriminator_model(optimizer)

    def _build_discriminator_model(self, optimizer):
        """
        @TODO
        Args:
            optimizer:

        Returns:

        """
        self.discriminator.compile(loss=[binary_crossentropy], optimizer=Adadelta(), metrics=['accuracy'])

    def _build_combined_model(self, optimizer):
        """
        @TODO
        Args:
            optimizer:

        Returns:

        """
        gen_input_latent = Input((self.latent_size,), name="Latent_Input")
        gen_image = self.generator(gen_input_latent)
        disc_image = self.discriminator(gen_image)
        self.combined_model = Model(gen_input_latent, disc_image)
        self.combined_model.compile(optimizer, loss=[binary_crossentropy])
        self.combined_model.n_epochs = 0

    def _build_generator(self, year: bool = False):
        """
        @TODO
        Returns:

        """
        noise_input = Input((self.latent_size,))
        if year:
            year_input = Input((1,))
            x = Concatenate()([noise_input, year_input])
            x = Dense(8192, name='Generator_Dense')(x)
        else:
            x = Dense(8192, name='Generator_Dense')(noise_input)
        x = Reshape((4, 4, 512))(x)

        x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding="same",
                   bias_initializer=initializers.zero(),
                   kernel_initializer=initializers.random_normal(stddev=1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = PixelNorm()(x)
        x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                   bias_initializer=initializers.zero(),
                   kernel_initializer=initializers.random_normal(stddev=1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = PixelNorm()(x)
        x = UpSampling2D((2, 2))(x)

        cur_img_size = 8
        n_kernels = 256
        while cur_img_size < self.img_shape[0]:
            x = Conv2D(n_kernels, kernel_size=(3, 3), strides=(1, 1), padding="same",
                       bias_initializer=initializers.zero(),
                       kernel_initializer=initializers.random_normal(stddev=1))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = PixelNorm()(x)
            x = Conv2D(n_kernels, kernel_size=(3, 3), strides=(1, 1), padding="same",
                       bias_initializer=initializers.zero(),
                       kernel_initializer=initializers.random_normal(stddev=1))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = PixelNorm()(x)
            x = UpSampling2D((2, 2))(x)
            cur_img_size *= 2
            n_kernels //= 2

        generator_output = Conv2D(self.channels, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                  bias_initializer=initializers.zero(),
                                  kernel_initializer=initializers.random_normal(stddev=1),
                                  activation="tanh")(x)
        if year:
            generator_model = Model([noise_input, year_input], generator_output)
        else:
            generator_model = Model(noise_input, generator_output)
        return generator_model

    def _build_discriminator(self, year: bool = False):
        """
        @TODO
        Returns:

        """
        image_input = Input(self.img_shape)
        if year:
            year_input = Input((1,))
        x = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer=initializers.zero(),
                   kernel_initializer=initializers.random_normal(stddev=1))(image_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = PixelNorm()(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', bias_initializer=initializers.zero(),
                   kernel_initializer=initializers.random_normal(stddev=1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = PixelNorm()(x)

        cur_img_size = self.img_shape[0] // 2
        n_kernels = 64
        while cur_img_size > 4:
            x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', bias_initializer=initializers.zero(),
                       kernel_initializer=initializers.random_normal(stddev=1))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = PixelNorm()(x)
            n_kernels *= 2
            cur_img_size //= 2

        x = Flatten()(x)
        x = Dense(self.latent_size)(x)
        if year:
            x = Concatenate()([x, year_input])
        discriminator_output = Dense(1)(x)
        if self.__class__ == CoverGAN:
            discriminator_output = Activation("sigmoid")(discriminator_output)
        if year:
            discriminative_model = Model([image_input, year_input], discriminator_output)
        else:
            discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def train_on_batch(self, real_images: np.array):
        """
        Runs a single gradient update on a single batch of data.

        Args:
            real_images: numpy array of real input images used for training

        Returns:
            In case of a simple GAN the current mean of the discriminator accurancy and the current mean of the
            binary crossentropy of the combined model is returned
        """
        fake = np.ones(len(real_images))
        real = np.zeros(len(real_images))

        noise = np.random.normal(size=(len(real_images), self.latent_size))
        generated_images = self.generator.predict(noise)

        discriminator_x = np.concatenate((generated_images, real_images))
        discriminator_y = np.concatenate((fake, real))
        self.discriminator_accuracy.append(self.discriminator.evaluate(discriminator_x, discriminator_y, verbose=0)[1])

        self.generator_loss.append([self.combined_model.train_on_batch(noise, real)])

        if np.mean(self.discriminator_accuracy) <= .5:
            self.train_discriminator(generated_images, real_images)

        return np.mean(self.discriminator_accuracy), np.mean(self.generator_loss)

    def train_discriminator(self, generated_images, real_images):
        """
        @TODO
        Args:
            generated_images:
            real_images:

        Returns:

        """
        fake = np.ones(len(generated_images))
        real = np.zeros(len(real_images))
        self.discriminator.train_on_batch(generated_images, fake)
        self.discriminator.train_on_batch(real_images, real)

    def _reset_generator_loss(self):
        """
        @TODO
        Returns:

        """
        self.generator_loss = []

    def _reset_discriminator_accuracy(self):
        """
        @TODO
        Returns:

        """
        self.discriminator_accuracy = []

    def reset_metrics(self):
        """
        @TODO
        Returns:

        """
        self._reset_generator_loss()
        self._reset_discriminator_accuracy()


class WGAN(CoverGAN):

    def __init__(self, batch_size, gradient_penalty_weight, **kwargs):
        """
        @TODO
        Args:
            img_height:
            img_width:
            channels:
            latent_size:
        """
        super(WGAN, self).__init__(**kwargs)
        self.discriminator_loss = []
        self.batch_size = batch_size
        self._gradient_penalty_weight = gradient_penalty_weight

    def build_models(self, optimizer=Adam(beta_1=0, beta_2=0.99), year: bool = False):
        """
        @TODO
        Builds the desired GAN that allows to generate covers.

        The GAN can either be a simple GAN or a WGAN using Wasserstein loss with gradient penalty to improve
        learning. Also additional information can be passed to the GAN like release year information.

        Args:
            optimizer: Which optimizer to use
            simple GAN with binary crossentropy loss
        """
        self.discriminator = self._build_discriminator(year)
        self.generator = self._build_generator(year)
        self.discriminator.trainable = False

        self._build_combined_model(optimizer, year)

        self.discriminator.trainable = True
        self.generator.trainable = False

        self._build_discriminator_model(optimizer, year)

    def _build_combined_model(self, optimizer, year: bool = False):
        """
        @TODO
        Args:
            optimizer:

        Returns:

        """
        gen_input_latent = Input((self.latent_size,), name="Latent_Input")
        if year:
            year_input = Input((1,))
            gen_image = self.generator([gen_input_latent, year_input])
            disc_image = self.discriminator([gen_image, year_input])
            self.combined_model = Model([gen_input_latent, year_input], disc_image)
        else:
            gen_image = self.generator(gen_input_latent)
            disc_image = self.discriminator(gen_image)
            self.combined_model = Model(gen_input_latent, disc_image)
        self.combined_model.compile(optimizer, loss=[wasserstein_loss])
        self.combined_model.n_epochs = 0

    def _build_discriminator_model(self, optimizer, year: bool = False):
        """
        @TODO
        Args:
            optimizer:

        Returns:

        """
        disc_input_image = Input(self.img_shape, name="Img_Input")
        disc_input_noise = Input((self.latent_size,), name="Noise_Input_for_Discriminator")
        year_input = Input((1,))
        if year:
            gen_image_disc = self.generator([disc_input_noise, year_input])
            disc_image_gen = self.discriminator([gen_image_disc, year_input])
            disc_image_image = self.discriminator([disc_input_image, year_input])
            avg_samples = RandomWeightedAverage(self.batch_size)([disc_input_image, gen_image_disc])
            disc_avg_disc = self.discriminator([avg_samples, year_input])
        else:
            gen_image_disc = self.generator(disc_input_noise)
            disc_image_gen = self.discriminator(gen_image_disc)
            disc_image_image = self.discriminator(disc_input_image)
            avg_samples = RandomWeightedAverage(self.batch_size)([disc_input_image, gen_image_disc])
            disc_avg_disc = self.discriminator(avg_samples)
        if year:
            self.discriminator_model = Model(inputs=[disc_input_image, disc_input_noise, year_input],
                                             outputs=[disc_image_image, disc_image_gen, disc_avg_disc])
        else:
            self.discriminator_model = Model(inputs=[disc_input_image, disc_input_noise],
                                             outputs=[disc_image_image, disc_image_gen, disc_avg_disc])
        partial_gp_loss = functools.partial(gradient_penalty_loss, averaged_samples=avg_samples,
                                            gradient_penalty_weight=self._gradient_penalty_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'
        self.discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                                         optimizer=optimizer)

    def train_on_batch(self, real_images, ratio: int = 5, year: np.array = None):
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

        if year is None:
            self.generator_loss.append([self.combined_model.train_on_batch(noise, real)])
            for _ in range(ratio):
                losses = self.discriminator_model.train_on_batch([real_images, noise], [real, fake, dummy_y])
        else:
            self.generator_loss.append([self.combined_model.train_on_batch([noise, year], real)])
            for _ in range(ratio):
                losses = self.discriminator_model.train_on_batch([real_images, noise, year], [real, fake, dummy_y])

        self.discriminator_loss.append(losses)

        return np.array(self.discriminator_loss).mean(axis=0), np.mean(self.generator_loss)

    def _reset_discriminator_loss(self):
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
        self._reset_generator_loss()
        self._reset_discriminator_accuracy()
        self._reset_discriminator_loss()
