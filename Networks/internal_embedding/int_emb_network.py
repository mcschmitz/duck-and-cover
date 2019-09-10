from keras import Model
from keras.layers import *
from keras.optimizers import Adam, Adadelta


class DACInternalGenreEmbedding:
    """
    @TODO
    """

    def __init__(self, img_height: int, img_width: int, n_genres: int, optimizer=Adam(0.0002, 0.5), channels: int = 3,
                 latent_size: int = 64, genre_prob: float = 0.05):
        """
        @TODO
        Args:
            img_height:
            img_width:
            n_genres:
            optimizer:
            channels:
            latent_size:
            genre_prob:
        """
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.n_genres = n_genres
        self.latent_size = latent_size
        self.genre_prob = genre_prob

        self.genre_embedder = self.build_genre_embedder()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=Adadelta(), metrics=['accuracy'])

        image_input = Input((self.latent_size,), name="Noise_Input")
        genre_input = Input((self.n_genres,), name="Genre_Input")

        self.generator = self.build_generator()

        generated_image = self.generator([image_input, genre_input])

        self.discriminator.trainable = False

        eval_of_gen_image = self.discriminator([generated_image, genre_input])
        self.adversarial_model = Model([image_input, genre_input], eval_of_gen_image)
        self.adversarial_model.compile(loss=['binary_crossentropy'], optimizer=optimizer)
        self.adversarial_model.n_epochs = 0
        self.discriminator_accuracy = []
        self.generator_loss = []

    def build_generator(self):
        """
        @TODO
        Returns:

        """
        noise_input = Input((self.latent_size,))
        genre_input = Input((self.n_genres,))
        embedded_genre = self.genre_embedder(genre_input)
        x = Concatenate()([noise_input, embedded_genre])
        x = Dense(8192, name='decoder_DENSE')(x)
        x = Reshape((4, 4, 512))(x)
        x = ReLU()(x)
        x = Deconv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_CONV1')(x)
        x = ReLU()(x)
        x = Deconv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_CONV2')(x)
        x = ReLU()(x)
        x = Deconv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_CONV3')(x)
        x = ReLU()(x)
        x = Deconv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_CONV4')(x)
        x = ReLU()(x)
        x = Deconv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_CONV5')(x)
        x = ReLU()(x)
        x = Deconv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_CONV6')(x)
        x = ReLU()(x)
        x = Deconv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        generator_output = Activation('tanh')(x)
        generator_model = Model([noise_input, genre_input], generator_output)
        return generator_model

    def build_discriminator(self):
        """
        @TODO
        Returns:

        """
        image_input = Input(self.img_shape)
        x = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV1')(image_input)
        x = LeakyReLU()(x)
        x = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV2')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV3')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV4')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV5')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV6')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        genre_input = Input((self.n_genres,))
        embedded_genre = self.genre_embedder(genre_input)
        x = Concatenate()([x, embedded_genre])

        discriminator_output = Dense(1, activation='sigmoid')(x)
        discriminative_model = Model([image_input, genre_input], discriminator_output)
        return discriminative_model

    def build_genre_embedder(self):
        """
        @TODO
        Returns:

        """
        genre_input = Input((self.n_genres,))
        emb1 = Dense(1024, activation='relu')(genre_input)
        emb2 = Dense(512, activation='relu')(emb1)
        embedded_genre = Dense(self.latent_size, activation='sigmoid')(emb2)
        genre_embedder = Model(genre_input, embedded_genre)
        return genre_embedder

    def train_on_batch(self, real_images, genre_input):
        """
        @TODO
        Args:
            real_images:
            genre_input:

        Returns:

        """
        fake = np.ones(len(genre_input))
        real = np.zeros(len(genre_input))

        noise = np.random.uniform(size=(len(genre_input), self.latent_size))
        genre_noise = np.random.choice([0, 1], size=(len(genre_input), self.n_genres),
                                       p=[1 - self.genre_prob, self.genre_prob])
        generated_images = self.generator.predict([noise, genre_noise])

        discriminator_x = np.concatenate((generated_images, real_images))
        discriminator_y = np.concatenate((fake, real))
        discriminator_genre_input = np.concatenate((genre_input, genre_input))
        self.discriminator_accuracy.append(
            self.discriminator.evaluate([discriminator_x, discriminator_genre_input],
                                        discriminator_y, verbose=0)[1])

        self.generator_loss.append([self.adversarial_model.train_on_batch([noise, genre_noise], real)])

        if np.mean(self.discriminator_accuracy) <= .5:
            self.train_discriminator(generated_images, real_images, genre_input, genre_noise)

        return np.mean(self.discriminator_accuracy), np.mean(self.generator_loss)

    def train_discriminator(self, generated_images, real_images, genre_input, genre_noise):
        """
        @TODO
        Args:
            generated_images:
            real_images:
            genre_input:
            genre_noise:

        Returns:

        """
        fake = np.ones(len(genre_input))
        real = np.zeros(len(genre_input))
        self.discriminator.train_on_batch([generated_images, genre_noise], fake)
        self.discriminator.train_on_batch([real_images, genre_input], real)
