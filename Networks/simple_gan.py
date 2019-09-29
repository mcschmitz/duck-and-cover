from keras import Model
from keras.layers import *
from keras.optimizers import Adam, Adadelta


class DaCSimple:

    def __init__(self, img_height: int, img_width: int, channels: int = 3, latent_size: int = 128):
        """
        @TODO
        Args:
            img_height:
            img_width:
            channels:
            latent_size:
        """
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_size = latent_size
        self.discriminator = None
        self.generator = None
        self.adversarial_model = None
        self.generator_loss = []
        self.discriminator_accuracy = []

    def build_models(self, optimizer=Adam(beta_1=0, beta_2=0.99)):
        """
        @TODO
        Args:
            optimizer:

        Returns:

        """
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=Adadelta(), metrics=['accuracy'])

        image_input = Input((self.latent_size,), name="Noise_Input")

        self.generator = self.build_generator()

        generated_image = self.generator(image_input)

        self.discriminator.trainable = False

        eval_of_gen_image = self.discriminator(generated_image)
        self.adversarial_model = Model(image_input, eval_of_gen_image)
        self.adversarial_model.compile(loss=['binary_crossentropy'], optimizer=optimizer)
        self.adversarial_model.n_epochs = 0

    def build_generator(self):
        """
        @TODO
        Returns:

        """
        noise_input = Input((self.latent_size,))
        x = Dense(8192, name='Generator_Dense')(noise_input)
        x = Reshape((4, 4, 512))(x)

        x = Conv2DTranspose(512, kernel_size=(4, 4), strides=(1, 1), padding="same",
                            bias_initializer=initializers.zero(),
                            kernel_initializer=initializers.random_normal(stddev=1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                            bias_initializer=initializers.zero(),
                            kernel_initializer=initializers.random_normal(stddev=1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)

        cur_img_size = 8
        n_kernels = 256
        while cur_img_size < self.img_shape[0]:
            x = Conv2DTranspose(n_kernels, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                bias_initializer=initializers.zero(),
                                kernel_initializer=initializers.random_normal(stddev=1))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(n_kernels, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                bias_initializer=initializers.zero(),
                                kernel_initializer=initializers.random_normal(stddev=1))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)
            x = UpSampling2D((2, 2))(x)
            cur_img_size *= 2
            n_kernels //= 2

        generator_output = Conv2DTranspose(3, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                           bias_initializer=initializers.zero(),
                                           kernel_initializer=initializers.random_normal(stddev=1),
                                           activation="tanh")(x)
        generator_model = Model(noise_input, generator_output)
        return generator_model

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

        discriminator_output = Dense(1, activation='sigmoid')(x)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def train_on_batch(self, real_images):
        """
        @TODO
        Args:
            real_images:

        Returns:

        """
        fake = np.ones(len(real_images))
        real = np.zeros(len(real_images))

        noise = np.random.normal(size=(len(real_images), self.latent_size))
        generated_images = self.generator.predict(noise)

        discriminator_x = np.concatenate((generated_images, real_images))
        discriminator_y = np.concatenate((fake, real))
        self.discriminator_accuracy.append(self.discriminator.evaluate(discriminator_x, discriminator_y, verbose=0)[1])

        self.generator_loss.append([self.adversarial_model.train_on_batch(noise, real)])

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

    def __reset_generator_loss(self):
        """
        @TODO
        Returns:

        """
        self.generator_loss = []

    def __reset_discriminator_accuracy(self):
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
        self.__reset_generator_loss()
        self.__reset_discriminator_accuracy()
