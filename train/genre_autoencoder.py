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

logger = pl.loggers.WandbLogger(
    project="duck-and-cover", entity="manne", tags=["genre-autoencoder"]
)
callbacks = [
    pl.callbacks.EarlyStopping(
        monitor="val/exact-match-ratio",
        mode="max",
        patience=int(TRAIN_STEPS * 0.1),
        verbose=True,
    ),
    pl.callbacks.ModelCheckpoint(
        monitor="val/exact-match-ratio",
        dirpath=model_dump_path,
        mode="max",
        verbose=True,
        save_last=False,
        every_n_train_steps=0,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    ),
]

trainer = pl.Trainer(
    gpus=-1,
    max_steps=TRAIN_STEPS,
    logger=logger,
    val_check_interval=100,
    enable_progress_bar=False,
)
trainer.fit(
    autoencoder,
    train_dataloader=data_loader.train_generator,
    val_dataloaders=data_loader.val_generator,
)
