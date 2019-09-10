from matplotlib import pyplot as plt

from Loader.internal_embedding.int_emb_loader import *
from Networks.internal_embedding.int_emb_network import CAAE

BATCH_SIZE = 32
EPOCH_NUM = 1000
STEPS_PER_EPOCH = 50
image_width = 602
image_height = 312
image_ratio = (1, 1)
image_size = 256

data_loader = ImageLoader(root="data/covers", batch_size=BATCH_SIZE, image_size=image_size, image_ratio=image_ratio)

image_width_compr = data_loader.image_shape[0]
image_height_compr = data_loader.image_shape[1]

caae = CAAE(img_width=image_width_compr, img_height=image_height_compr, latent_size=64)

g_losses = []
d_accs_img = []
d_accs_z = []

for epoch in range(0, EPOCH_NUM):
    for step in range(0, STEPS_PER_EPOCH):
        step += 1
        print('Running Batch {} of {} of Epoch {}'.format(step, STEPS_PER_EPOCH, caae.adversarial_model.n_epochs + 1))
        new_images, old_images, erosion = data_loader.next()
        cum_g_loss, cum_d_acc_img, cum_d_acc_z = caae.train_on_batch(new_images, old_images, erosion)
    caae.adversarial_model.n_epochs += 1
    img = data_loader.next(specific_class="Eur5Back")[0][0]
    img = array_to_img(img)
    final_image = pil_image.new("RGB", (image_width_compr * 6, image_height_compr * 2))
    x_offset = 0
    final_image.paste(img, (x_offset, 0))
    img = img_to_array(img).reshape((1, image_height_compr, image_width_compr, 3)) / 255
    erosion = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for e in erosion:
        x_offset += image_width_compr
        predicted_img = caae.generator.predict([img, np.array(e).reshape(-1, 1)])
        predicted_img = predicted_img.reshape(image_height_compr, image_width_compr, 3)
        predicted_img = array_to_img(predicted_img)
        final_image.paste(predicted_img, (x_offset, 0))
    img = data_loader.next(specific_class="Eur5Front")[0][0]
    img = array_to_img(img)
    x_offset = 0
    final_image.paste(img, (x_offset, image_height_compr))
    img = img_to_array(img).reshape((1, image_height_compr, image_width_compr, 3)) / 255
    erosion = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for e in erosion:
        x_offset += 256
        predicted_img = caae.generator.predict([img, np.array(e).reshape(-1, 1)])
        predicted_img = predicted_img.reshape(image_height_compr, image_width_compr, 3)
        predicted_img = array_to_img(predicted_img)
        final_image.paste(predicted_img, (x_offset, image_width_compr))
    prediction_path = 'learning_progress/epoch{}.png'.format(caae.adversarial_model.n_epochs)
    final_image.save(prediction_path)

    g_losses.append(cum_g_loss)
    fig0 = plt.figure(0)
    ax = fig0.add_subplot(1, 1, 1)
    ax.plot(list(range(caae.adversarial_model.n_epochs)), g_losses)
    plt.ylabel('Generator MSE')
    plt.xlabel('Epochs')
    ax.set_yscale('log')
    plt.savefig('learning_progress/g_loss.png')
    plt.close()

    d_accs_img.append(cum_d_acc_img)
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(list(range(caae.adversarial_model.n_epochs)), d_accs_img)
    plt.ylabel('Discriminator Img Accuracy')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/d_loss_img.png')
    plt.close()

    d_accs_z.append(cum_d_acc_z)
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(list(range(caae.adversarial_model.n_epochs)), d_accs_z)
    plt.ylabel('Discriminator z Accucary')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/d_loss_z.png')
    plt.close()

    if epoch % 1 == 0:
        img = load_img('10mint.png', target_size=(image_width_compr, image_height_compr))
        img = img_to_array(img)
        img = img.reshape(image_height_compr, image_width_compr, 3) / 255
        final_image = pil_image.new("RGB", (image_width_compr * 6, 256))
        x_offset = 0
        final_image.paste(array_to_img(img), (x_offset, 0))
        img = img.reshape(1, image_height_compr, image_width_compr, 3)
        for e in erosion:
            x_offset += 256
            predicted_img = caae.generator.predict([img, np.array(e).reshape(-1, 1)])
            predicted_img = predicted_img.reshape(256, image_width_compr, 3)
            predicted_img = array_to_img(predicted_img)
            final_image.paste(predicted_img, (x_offset, 0))
        prediction_path = 'learning_progress/10_epoch{}.png'.format(caae.adversarial_model.n_epochs)
        final_image.save(prediction_path)
