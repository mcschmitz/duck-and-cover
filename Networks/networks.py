import functools
from abc import abstractmethod, ABC

import numpy as np
from keras import Model
from keras.layers import *
from keras.losses import binary_crossentropy

from Networks.utils import wasserstein_loss, RandomWeightedAverage, gradient_penalty_loss


#  TODO Add Release Year information
#  TODO Add Genre Information
#  TODO Let GAN grow
#  TODO Try styleGAN

class GAN(ABC):
    def __init__(self):
        """Abstract GAN Class
        """
        self.discriminator = None
        self.generator = None
        self.discriminator_model = None
        self.combined_model = None
        self.images_shown = 0
        self.generator_loss = []
        self.history = {}

    @abstractmethod
    def build_models(self, *args):
        """Abstract method to build the models for the GAN
        """
        pass

    @abstractmethod
    def _build_discriminator(self):
        """Abstract method to define the discriminator architecture
        """
        pass

    @abstractmethod
    def _build_discriminator_model(self, *args):
        """Abstract method to build the discriminator
        """
        pass

    @abstractmethod
    def _build_combined_model(self, *args):
        """Abstract method to build the combined model
        """
        pass

    @abstractmethod
    def _build_generator(self):
        """Abstract method to define the generator architecture
        """
        pass

    @abstractmethod
    def train_on_batch(self, *args):
        """Abstract method to train the combined model on a batch of data
        """
        pass

    @abstractmethod
    def train_discriminator(self, *args):
        """Abstract method to train the discriminator on a batch of data
        """
        pass


class CoverGAN(GAN):

    def __init__(self, img_height: int, img_width: int, channels: int = 3, latent_size: int = 128):
        """Simple Deep Convolutional GAN

        Simple DCGAN that builds a discriminator a generator and the adversarial model to train the GAN based on the
        binary crossentropy loss for the generator and the discriminator

        Args:
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the image
        """
        super(CoverGAN, self).__init__()
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_size = latent_size
        self.discriminator_accuracy = []
        self.critic_below50 = 0

    def build_models(self, combined_optimizer, discriminator_optimizer=None):
        """Builds the desired GAN that allows to generate covers.

        Creates every model needed for GAN. Creates a discriminator, a generator and the combined model. The
        discriminator as well as the generator are trained using the provided optimizer.

        Args:
            combined_optimizer: Which optimizer to use for the combined model
            discriminator_optimizer: Which optimizer to use for the discriminator model
        """
        discriminator_optimizer = combined_optimizer if discriminator_optimizer is None else discriminator_optimizer
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.discriminator.trainable = False
        self._build_combined_model(combined_optimizer)
        self.history["G_loss"] = []

        self.generator.trainable = False
        self.discriminator.trainable = True
        self._build_discriminator_model(discriminator_optimizer)
        self.history["D_accuracy"] = []

    def _build_discriminator_model(self, optimizer):
        """Build the discriminator model

        The discriminator model is only compiled in this step since the model is rather simple in a plain GAN

        Args:
            optimizer: Which optimizer to use
        """
        self.discriminator.compile(loss=[binary_crossentropy], optimizer=optimizer, metrics=['accuracy'])

    def _build_combined_model(self, optimizer):
        """Build the combined GAN consisting of generator and discriminator

        Takes the latent input and generates an images out of it by applying the generator. Classifies the image by
        via the discriminator. The model is compiled using the given optimizer

        Args:
            optimizer: Which optimizer to use
        """
        gen_input_latent = Input((self.latent_size,), name="Latent_Input")
        gen_image = self.generator(gen_input_latent)
        disc_image = self.discriminator(gen_image)
        self.combined_model = Model(gen_input_latent, disc_image)
        self.combined_model.compile(optimizer, loss=[binary_crossentropy])

    def _build_generator(self):
        """Builds the DCGAN generator

        Builds the very simple generator that takes a latent input vector and applies the following block until the
        desired image size is reached: 3x3 convolutional layer with ReLu activation -> Batch Normalization ->
        Upsamling layer. The last Convolutional layer wit tanH activation results in 3 RGB channels and serves as
        final output

        Returns:
            The DCGAN generator
        """
        noise_input = Input((self.latent_size,))
        x = Dense(8192, name='Generator_Dense')(noise_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((4, 4, 512))(x)

        cur_img_size = 4
        n_kernels = 256
        while cur_img_size < self.img_shape[0]:
            x = Conv2D(n_kernels, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
            x = UpSampling2D()(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            cur_img_size *= 2
            n_kernels //= 2

        generator_output = Conv2D(self.channels, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        generator_output = Activation("tanh")(generator_output)
        generator_model = Model(noise_input, generator_output)
        return generator_model

    def _build_discriminator(self):
        """Builds the DCGAN discriminator

        Builds the very simple discriminator that takes an image input and applies a 3x3 convolutional layer with ReLu
        activation and a 2x2 stride until the desired embedding  size is reached. The flattend embedding is ran
        through a Dense layer with sigmoid output to label the image.

        Returns:
            The DCGAN discriminator
        """
        image_input = Input(self.img_shape)
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(image_input)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        cur_img_size = self.img_shape[0] // 2
        n_kernels = 64
        while cur_img_size > 4:
            x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)
            n_kernels *= 2
            cur_img_size //= 2

        x = Flatten()(x)
        discriminator_output = Dense(1)(x)
        discriminator_output = Activation("sigmoid")(discriminator_output)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def train_on_batch(self, real_images: np.array):
        """Runs a single gradient update on a batch of data.

        Args:
            real_images: numpy array of real input images used for training
        """
        fake = np.ones(len(real_images))
        real = np.zeros(len(real_images))

        noise = np.random.normal(size=(len(real_images), self.latent_size))
        generated_images = self.generator.predict(noise)

        discriminator_x = np.concatenate((generated_images, real_images))
        discriminator_y = np.concatenate((fake, real))

        self.train_discriminator(generated_images, real_images)
        discriminator_batch_acc = self.discriminator.evaluate(discriminator_x, discriminator_y, verbose=0)[1]
        self.history["D_accuracy"].append(discriminator_batch_acc)

        self.history["G_loss"].append(self.combined_model.train_on_batch(noise, real))

    def train_discriminator(self, generated_images: np.array, real_images: np.array):
        """Runs a single gradient update for the discriminator

        Args:
            generated_images: array of generated images by the generator
            real_images: array of real images
        """
        fake = np.ones(len(generated_images))
        real = np.zeros(len(real_images))
        self.discriminator.train_on_batch(generated_images, fake)
        self.discriminator.train_on_batch(real_images, real)


class WGAN(GAN):

    def __init__(self, batch_size, gradient_penalty_weight, img_height: int, img_width: int, channels: int = 3,
                 latent_size: int = 128):
        """
        @TODO
        Args:
            img_height:
            img_width:
            channels:
            latent_size:
        """
        super(WGAN, self).__init__()
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_size = latent_size
        self.discriminator_loss = []
        self.batch_size = batch_size
        self._gradient_penalty_weight = gradient_penalty_weight

    def build_models(self, optimizer):
        """
        @TODO
        Builds the desired GAN that allows to generate covers300.

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
        self.history["G_loss"] = []

        self.generator.trainable = False
        self.discriminator.trainable = True
        self._build_discriminator_model(optimizer)
        self.history["D_loss_positives"] = []
        self.history["D_loss_negatives"] = []
        self.history["D_loss_dummies"] = []

    def _build_discriminator(self):
        """Builds the WGAN-GP discriminator

        Builds the discriminator that takes an image input and applies a 3x3 convolutional layer with LeakyReLu
        activation and a 2x2 stride until the desired embedding  size is reached. The flattend embedding is ran
        through a 1024 Dense layer followed by a output layer with one linear output node.

        Returns:
            The DCGAN discriminator
        """
        image_input = Input(self.img_shape)
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(image_input)
        x = LeakyReLU()(x)

        cur_img_size = self.img_shape[0] // 2
        n_kernels = 64
        while cur_img_size > 4:
            x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)
            x = LeakyReLU()(x)
            n_kernels *= 2
            cur_img_size //= 2

        x = Flatten()(x)
        x = Dense(self.latent_size)(x)
        x = LeakyReLU()(x)
        x = Dense(1024, kernel_initializer='he_normal')(x)
        discriminator_output = Dense(1, kernel_initializer='he_normal')(x)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def _build_discriminator_model(self, optimizer):
        """
        @TODO
        Args:
            optimizer:

        Returns:

        """
        disc_input_image = Input(self.img_shape, name="Img_Input")
        disc_input_noise = Input((self.latent_size,), name="Noise_Input_for_Discriminator")
        gen_image_disc = self.generator(disc_input_noise)
        disc_image_gen = self.discriminator(gen_image_disc)
        disc_image_image = self.discriminator(disc_input_image)
        avg_samples = RandomWeightedAverage(self.batch_size)([disc_input_image, gen_image_disc])
        disc_avg_disc = self.discriminator(avg_samples)
        self.discriminator_model = Model(inputs=[disc_input_image, disc_input_noise],
                                         outputs=[disc_image_image, disc_image_gen, disc_avg_disc])
        partial_gp_loss = functools.partial(gradient_penalty_loss, averaged_samples=avg_samples,
                                            gradient_penalty_weight=self._gradient_penalty_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'
        self.discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                                         optimizer=optimizer)

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
        self.combined_model.compile(optimizer, loss=[wasserstein_loss])

    def _build_generator(self):
        """Builds the DCGAN generator
        @TODO
        Builds the very simple generator that takes a latent input vector and applies the following block until the
        desired image size is reached: 3x3 convolutional layer with ReLu activation -> Batch Normalization ->
        Upsamling layer. The last Convolutional layer wit tanH activation results in 3 RGB channels and serves as
        final output

        Returns:
            The DCGAN generator
        """
        noise_input = Input((self.latent_size,))
        x = Dense(1024)(noise_input)
        x = LeakyReLU(.2)(x)
        x = Dense(8192)(x)
        x = LeakyReLU(.2)(x)
        x = BatchNormalization()(x)
        x = Reshape((4, 4, 512))(x)

        cur_img_size = 4
        n_kernels = 256
        while cur_img_size < self.img_shape[0]:
            x = Conv2D(n_kernels, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
            x = UpSampling2D()(x)
            x = LeakyReLU(.2)(x)
            x = BatchNormalization()(x)
            cur_img_size *= 2
            n_kernels //= 2

        generator_output = Conv2D(self.channels, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        generator_output = Activation("tanh")(generator_output)
        generator_model = Model(noise_input, generator_output)
        return generator_model

    def train_on_batch(self, real_images, n_critic: int = 5):
        """
        @TODO
        Args:
            real_images:
            n_critic:

        Returns:

        """
        batch_size = real_images.shape[0] // n_critic
        real_y = np.ones((batch_size, 1))

        for i in range(n_critic):
            discriminator_minibatch = real_images[i * batch_size:(i + 1) * batch_size]
            losses = self.train_discriminator(discriminator_minibatch, real_y)

        self.discriminator_loss.append(losses)

        noise = np.random.normal(size=(batch_size, self.latent_size))
        self.generator_loss.append([self.combined_model.train_on_batch(noise, real_y)])

        return np.array(self.discriminator_loss).mean(axis=0), np.mean(self.generator_loss)

    def train_discriminator(self, real_images, real_y):
        """
        @TODO
        Args:
            real_images:
            real_y:

        Returns:

        """
        batch_size = len(real_images)
        fake_y = np.ones((batch_size, 1)) * -1
        real_y = np.ones((batch_size, 1))
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(size=(batch_size, self.latent_size))
        losses = self.discriminator_model.train_on_batch([real_images, noise], [real_y, fake_y, dummy_y])
        return losses
