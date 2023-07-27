# **Duck and Cover**

`duck_and_cover` allows you to create your own album covers based on additional information like the genre of the LP, the release year and the name and titles of your artificial album.

It uses data from more than 600.000 covers of over 120.000 top artists on spotify from 3.254 genres to learn about the structure and appearance of let's say a thrash metal album cover from 1988.

## ⚙️ Setup

The project is built with Python 3.10 and uses `poetry` for dependency management. To install all dependencies simply run:

```shell
poetry install
```

### Data Gathering

As spotify prohibits the publication of their data, you have to gather the data to train the model yourself.

Data gathering consists of two steps:

1. Create a `.env` file with your `SPOTIPY_CLIENT_ID` and
   `SPOTIPY_CLIENT_SECRET` (Spotify API Client ID & Secret). Read more on
   how to get your Spotify client ID and secret [here](https://developer.spotify.com/documentation/general/guides/app-settings/).
  
2. Run the [data collection script](data_collection/collect_artist_data.py) which iteratively collects the top 50 artists for each of the genres listed in [this file](data/genres.txt) and their related artists whereby duplicated artists are removed. This script creates a table containing genre and release date of each album released by these artists, the artists and the album name as well as an URL to download a 300x300 as well as a 64x64 image of the cover. Based on this URL the covers are finally downloaded and save to a unified identifiable file structure. All of the final and intermediate results of the tasks in steps in 2. are saved in a temporary dictionary to allow splitting the data collection in case of reaching the quota limit of the Spotify API (which is usually not the case) or running into other trouble.

3. Once all the data is available you can run the [script](data_collection/add_caption_and_upload_to_hf.py), that adds the captions to the images and uploads them to the HuggingFace dataset hub.

4. Additionally you can run a script that collects example albums based on their album ID and saves them to a folder. This is useful for testing the model on a specific album. The script can be found [here](data_collection/collect_example_albums.py).

### Networks and results

The first network built is a simple [Deep Convolutional GAN](https://arxiv.org/pdf/1511.06434.pdf). The results aren't really satisfying since a DCGAN is not able to capture
the manifold variations in an album Cover and collapses pretty early on:

<div align="center">
  <img src="img/learning_progress_gan.gif">
</div>

Switching from a normal binary crossentropy loss for both discriminator and
the combined model to a [GAN trained with wasserstein loss fused with
gradient penalty](https://arxiv.org/pdf/1704.00028.pdf) yields much better results than the DCGAN:

<div align="center">
  <img src="img/learning_progress_wgan.gif">
</div>

Obviously optimizing the wasserstein loss results in more stable gradients which leads to a steady learning phase, whereas the gradient penalty prevents varnished gradients. This results in a detectable structure in the generated images, so that they even adumbrate interpret or album names on top or bottom of the generated covers.

Now let's have a look at the results of the ProGAN. Clearly once can see how the model is built up from a very small resolution to a final resolution of 512x512 pixel. This allows the network to learn the structure of the image little by little and produces an image that has clearer edges on those structures. The results are far from perfect, but much better than on the Deep Convolutional GAN and on the Wassertrein GAN.

<div align="center">
  <img src="img/learning_progress_progan.gif">
</div>

### Next Steps

1. Train ProGan + Genre
2. Integrate Artist Name
3. Integrate Album Name
4. Migrate W-GAN to PTLightning
5. Migrate DCGAN to PTLightning
