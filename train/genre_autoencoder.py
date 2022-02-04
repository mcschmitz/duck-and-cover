import os
from pathlib import Path

import pytorch_lightning as pl
from transformers import BertConfig

from config import config
from loader import GenreDataLoader
from networks import GenreAutoencoder, GenreDecoder, GenreEncoder

BATCH_SIZE = 64
LATENT_SIZE = 768
PATH = f"genre_autoencoder-{LATENT_SIZE}"
lp_path = os.path.join(config.get("learning_progress_path"), PATH)
TRAIN_STEPS = int(1e6)

model_dump_path = os.path.join(PATH, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)
Path(lp_path).mkdir(parents=True, exist_ok=True)


data_loader = GenreDataLoader(
    batch_size=BATCH_SIZE,
    meta_data_path="data/album_data_frame.json",
    tokenizer_path="data/genre_encoder_tokenizer",
)
num_labels = data_loader.get_number_of_classes()
vocab_size = data_loader.tokenizer.vocab_size

bert_config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=LATENT_SIZE,
    max_position_embeddings=64,
)

encoder = GenreEncoder(bert_config)

decoder = GenreDecoder(input_dim=LATENT_SIZE, num_labels=num_labels)

autoencoder = GenreAutoencoder(encoder, decoder)

trainer = pl.Trainer(gpus=-1, max_steps=TRAIN_STEPS)
trainer.fit(autoencoder, train_dataloader=data_loader.train_generator)
