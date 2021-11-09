import os
from functools import partial
from pathlib import Path

from transformers.optimization import get_linear_schedule_with_warmup

from loader import GenreDataLoader
from networks import GenreAutoencoder

BATCH_SIZE = 16
LATENT_SIZE = 128
PATH = f"genre_autoencoder-{LATENT_SIZE}"
TRAIN_STEPS = int(1e6)

model_dump_path = os.path.join(PATH, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)


scheduler = learning_rate_schedule = partial(
    get_linear_schedule_with_warmup,
    num_warmup_steps=int(TRAIN_STEPS * 0.1),
    num_training_steps=TRAIN_STEPS,
)

data_loader = GenreDataLoader(
    batch_size=BATCH_SIZE,
    meta_data_path="data/album_data_frame.json",
    tokenizer_path="data/genre_encoder_tokenizer",
)
vocab_size = data_loader.tokenizer.vocab_size

autoencoder = GenreAutoencoder(
    num_labels=3319,
    encoding_dim=LATENT_SIZE,
    scheduler=scheduler,
    optimizer_params={"lr": 0.001, "betas": (0.0, 0.99)},
    vocab_size=vocab_size,
)

autoencoder.train(
    data_loader=data_loader,
    steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    write_model_to=model_dump_path,
)
autoencoder.save(model_dump_path)
