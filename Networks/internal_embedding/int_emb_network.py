import keras.backend as K
import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import Adam, Adadelta


class CAAE:
    def __init__(self, img_height, img_width, optimizer=Adam(0.0002, 0.5), channels=3, latent_size=64):
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_size = latent_size
        self.greatest_common_divisor = 2 ** 6

        self.discriminative_model_img = self.build_discriminative_model_img()
        self.discriminative_model_img.compile(loss=['binary_crossentropy'], optimizer=Adadelta(), metrics=['accuracy'])

        self.discriminative_model_z = self.build_discriminative_model_z()
        self.discriminative_model_z.compile(loss=['binary_crossentropy'], optimizer=Adadelta(), metrics=['accuracy'])

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        image_input = Input(self.img_shape, name="Image_Input")
        erosion_input = Input((1,), name="Erosion_Input")
        encoded_image = self.encoder(image_input)
        reconstructed_image = self.decoder([encoded_image, erosion_input])

        self.generator = Model([image_input, erosion_input], [reconstructed_image])

        self.discriminative_model_img.trainable = False
        self.discriminative_model_z.trainable = False

        eval_of_gen_image = self.discriminative_model_img([reconstructed_image, erosion_input])
        eval_of_gen_z = self.discriminative_model_z(encoded_image)
        self.adversarial_model = Model([image_input, erosion_input],
                                       [reconstructed_image, eval_of_gen_image, eval_of_gen_z])
        self.adversarial_model.compile(loss=['mse', 'binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer,
                                       loss_weights=[0.999, 0.0005, 0.0005])
        self.adversarial_model.n_epochs = 0
        self.reconstruction_loss = []
        self.discriminator_img_accuracy = []
        self.discriminator_z_accuracy = []

    def build_encoder(self):
        image_input = Input(self.img_shape)
        x = Conv2D(16, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same', name='encoder_CONV1')(
            image_input)
        x = Conv2D(32, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same', name='encoder_CONV2')(x)
        x = Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same', name='encoder_CONV3')(x)
        x = Conv2D(128, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same', name='encoder_CONV4')(x)
        x = Conv2D(256, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same', name='encoder_CONV5')(x)
        x = Conv2D(512, kernel_size=(5, 5), activation='relu', strides=(2, 2), padding='same', name='encoder_CONV6')(x)
        x = Flatten()(x)
        encoder_output = Dense(self.latent_size, activation='relu')(x)
        encoder_model = Model(image_input, encoder_output)
        return encoder_model

    def build_decoder(self):
        encoded_input = Input((self.latent_size,))
        erosion_input = Input((1,))
        x = Concatenate()([encoded_input, erosion_input])
        x = Dense(8192, name='decoder_DENSE')(x)
        x = Reshape((4, 4, 512))(x)
        x = ReLU()(x)
        x = Deconv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_CONV1')(x)
        x = ReLU()(x)
        x = Deconv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_CONV2')(x)
        x = ReLU()(x)
        x = Deconv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_CONV3')(x)
        x = ReLU()(x)
        x = Deconv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_CONV4')(x)
        x = ReLU()(x)
        x = Deconv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_CONV5')(x)
        x = ReLU()(x)
        x = Deconv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_CONV6')(x)
        x = ReLU()(x)
        x = Deconv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        generator_output = Activation('tanh')(x)
        generator_model = Model([encoded_input, erosion_input], generator_output)
        return generator_model

    def build_discriminative_model_img(self):
        image_input = Input(self.img_shape)
        x = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', name='discriminator_CONV1')(image_input)
        x = LeakyReLU()(x)
        x_shape = (x.shape[1].value, x.shape[2].value)
        size = x_shape[0] * x_shape[1]

        def repeat(x):
            return K.repeat(x, size)

        erosion_input = Input((1,))
        erosion = Lambda(repeat)(erosion_input)
        erosion = Reshape((int(x_shape[0]), int(x_shape[1]), 1))(erosion)
        x = Concatenate()([x, erosion])
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
        discriminator_output = Dense(1, activation='sigmoid')(x)
        discriminative_model = Model([image_input, erosion_input], discriminator_output)
        return discriminative_model

    def build_discriminative_model_z(self):
        encoded_input = Input((self.latent_size,))
        x = Dense(64)(encoded_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(16)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        discriminator_z_output = Dense(1, activation='sigmoid')(x)
        discriminative_model_z = Model(encoded_input, discriminator_z_output)
        return discriminative_model_z

    def train_on_batch(self, image_input, image_output, erosion_input):
        fake = np.ones(len(erosion_input))
        valid = np.zeros(len(erosion_input))
        encoded_images = self.encoder.predict(image_input)
        generated_images = self.decoder.predict([encoded_images, erosion_input])

        discriminator_img_x = np.concatenate((generated_images, image_output))
        discriminator_y = np.concatenate((fake, valid))
        discriminator_img_erosion_input = np.concatenate((erosion_input, erosion_input))
        self.discriminator_img_accuracy.append(
            self.discriminative_model_img.evaluate([discriminator_img_x, discriminator_img_erosion_input],
                                                   discriminator_y, verbose=0)[1])

        if np.mean(self.discriminator_img_accuracy) <= .5:
            self.train_discriminator_img(generated_images, image_output, erosion_input)

        n_uniform_noise = self.latent_size * len(erosion_input)
        uniform_noise = np.random.uniform(0, 1, size=n_uniform_noise).reshape((len(erosion_input), self.latent_size))
        self.discriminator_z_accuracy.append(
            self.discriminative_model_z.evaluate(np.concatenate((uniform_noise, encoded_images)), discriminator_y,
                                                 verbose=0)[1])

        if np.mean(self.discriminator_z_accuracy) <= .5:
            self.train_discriminator_z(encoded_images, erosion_input)

        reconstruction_loss = \
        self.adversarial_model.train_on_batch([image_input, erosion_input], [image_output, valid, valid])[0]

        return reconstruction_loss, np.mean(self.discriminator_img_accuracy), np.mean(self.discriminator_z_accuracy)

    def train_discriminator_z(self, encoded_images, erosion_input):
        fake = np.ones(len(erosion_input))
        valid = np.zeros(len(erosion_input))
        n_uniform_noise = self.latent_size * len(erosion_input)
        uniform_noise = np.random.uniform(0, 1, size=n_uniform_noise).reshape((len(erosion_input), self.latent_size))
        self.discriminative_model_z.train_on_batch(uniform_noise, fake)
        self.discriminative_model_z.train_on_batch(encoded_images, valid)

    def train_discriminator_img(self, generated_images, image_output, erosion_input):
        fake = np.ones(len(erosion_input))
        valid = np.zeros(len(erosion_input))
        self.discriminative_model_img.train_on_batch([generated_images, erosion_input], fake)
        self.discriminative_model_img.train_on_batch([image_output, erosion_input], valid)
