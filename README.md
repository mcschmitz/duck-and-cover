# **Duck and Cover**

<div align="center">
  <img src="http://cbarks.dk/Digital/seraa196208.JPG">
</div>

**Duck and Cover allows you to create your own album covers based on
additional information like the genre and the release year of your
artificial performer.**

Duck and Cover uses data from more than 600.000 covers of over 120.000
spotify top artists from 3.254 genres to learn about the structure and
appearance of let's say a thrash metal album cover from 1988.

### Data Gathering
Data gathering consists of two steps:
1. Rename [credentials_template.py](credentials_template.py) to
   credentials.py and fill your spotify client ID and your Spotify
   client secret. Read more on how to get your Spotify client ID and
   secret
   [here](https://developer.spotify.com/documentation/general/guides/app-settings/).
2. Run the [data collection script](collect_artist_data.py) which
   iteratively collects the top 50 artists for each of the genres listed
   in [this file](data/genres.txt) and their related artists whereby
   duplicated artists are removed. After this step the script builds a
   table containing genre and release date of each album released by
   these artists, the artists and the album name as well as an URL to
   download a 300x300 as well as a 64x64 image of the cover. Based on 
   this URL the covers are finally downloaded and save to a unified
   identifiable file structure.
   
 All of the final and intermediate results of the tasks in steps in 2.
 are saved in a tmp dictionary to allow splitting the data collection in
 case of reaching the quota limit of the Spotify API (which is usually
 not the case) or running into other trouble.