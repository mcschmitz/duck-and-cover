import os
from pathlib import Path

import pytorch_lightning as pl
from transformers import BertConfig, BertForMaskedLM, BertModel

from config import config
from loader import GenreDataLoader
from networks import GenreAutoencoder, GenreDecoder

bert_config_name = "prajjwal1/bert-mini"

BATCH_SIZE = 64
bert_config = BertConfig.from_pretrained(bert_config_name)
PATH = f"genre_autoencoder-{bert_config.hidden_size}"
lp_path = os.path.join(config.get("learning_progress_path"), PATH)

TRAIN_STEPS = int(1e6)

model_dump_path = os.path.join(lp_path, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)

data_loader = GenreDataLoader(
    batch_size=BATCH_SIZE,
    meta_data_path="data/album_data_frame.json",
    tokenizer_path=bert_config_name,
)
num_labels = data_loader.get_number_of_classes()
vocab_size = len(data_loader.tokenizer.get_vocab())

encoder = BertForMaskedLM(bert_config)
encoder.bert = BertModel(bert_config, add_pooling_layer=True)
decoder = GenreDecoder(
    input_dim=bert_config.hidden_size, num_labels=num_labels
)
autoencoder = GenreAutoencoder(encoder, decoder)

logger = pl.loggers.WandbLogger(
    project="duck-and-cover", entity="mcschmitz", tags=["genre-autoencoder"]
)
total_number_of_validations = TRAIN_STEPS / len(data_loader.train_generator)
callbacks = [
    pl.callbacks.EarlyStopping(
        monitor="val/accuracy",
        mode="max",
        patience=int(total_number_of_validations * 0.1),
        verbose=True,
    ),
    pl.callbacks.ModelCheckpoint(
        monitor="val/accuracy",
        dirpath=model_dump_path,
        mode="max",
        verbose=True,
        save_last=False,
        every_n_train_steps=0,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    ),
    pl.callbacks.LearningRateMonitor(logging_interval="step"),
]
trainer = pl.Trainer(
    gpus=-1,
    max_steps=TRAIN_STEPS,
    logger=logger,
    val_check_interval=1.0,
    enable_progress_bar=False,
    weights_save_path=model_dump_path,
    callbacks=callbacks,
)
trainer.fit(
    autoencoder,
    train_dataloader=data_loader.train_generator,
    val_dataloaders=data_loader.val_generator,
)
model = autoencoder.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    encoder=encoder,
    decoder=decoder,
)

trainer.test(autoencoder, test_dataloaders=data_loader.test_generator)
