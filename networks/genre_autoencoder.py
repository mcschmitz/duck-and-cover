from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import hamming_loss
from torch import nn
from torch.optim import AdamW
from transformers import BatchEncoding, BertConfig, BertModel
from transformers.optimization import get_linear_schedule_with_warmup


class GenreDecoder(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        """
        Genre classification head. Takes the genre embedding as input and tries
        to classify the true genres. The.

        Args:
            input_dim: Latent size dimension
            num_labels: Total number of genres
        """
        super(GenreDecoder, self).__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction head that takes the genre embeddings as input and predicts
        the genre outputs.

        Args:
            x: Genre embeddings

        Returns:
            Predicted genres
        """
        x = self.classifier(x)
        return nn.Sigmoid()(x)


class GenreEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        """
        Genre encoderEncoder that uses BERT to encode a list of genres into a
        dense space.

        Args:
             config: Huggingface Bert Model config.
        """
        super(GenreEncoder, self).__init__()
        self.encoder = BertModel(config)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        Encoding method.

        Args:
            x: Tokenized genre input
        """
        return self.encoder(**x).pooler_output


class GenreAutoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        """
        A model which can be seen as an autoencoder (AE) for the genre strings.
        Takes a Bert-like tokenized concatenated string of genre strings as
        input and aims to predict the input genres from the pooled output of a
        small Bert-like model.

        Args:
            encoder: Genre encoder
            decoder: Genre decoder
        """
        super(GenreAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss = nn.BCELoss()

    def forward(self, x: BatchEncoding) -> torch.Tensor:
        """
        Runs a forward pass for a concatenated list of genre strings.

        Args:
            x: Bert-like batch encooding

        Returns:
            Genre embedding
        """
        encoding = self.encoder(x)
        return self.decoder(encoding)

    def configure_optimizers(self):
        """
        Assign both the discriminator and the generator optimizer.
        """
        encoders = (self.encoder, self.decoder)
        optimizer_grouped_parameters = []
        for encoder in encoders:
            params = [
                {
                    "params": [
                        p for p in encoder.parameters() if p.requires_grad
                    ],
                    "weight_decay": 0,
                },
            ]
            optimizer_grouped_parameters.extend(params)
        optimizer = AdamW(
            params=optimizer_grouped_parameters, lr=1e-6, betas=(0, 0.99)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.max_steps * 0.1),
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: BatchEncoding):
        """
        Trains the AE on the given batch.

        Args:
            batch: Tokenized genre sequence

        Returns:
            Scalar loss of this batch
        """
        x, labels = batch
        x = self(x)
        loss = self.loss(x, labels)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple, _batch_idx) -> Dict:
        """
        Takes a batch from the validation generator and predicts it.

        Args:
            batch: Batch. A Tuple of data and labels
            _batch_idx: Batch index
        """
        data, labels = batch
        return {"pred": self(data), "true": labels}

    def validation_epoch_end(self, x: List[Dict]):
        """
        Takes the predictions from the validation steps and calculates the
        validation metrics on it.

        Args:
            x: List of validation_step outputs
        """
        prediction = torch.cat([xi["pred"].cpu() for xi in x])
        true = torch.cat([xi["true"].cpu() for xi in x])
        loss = self.loss(prediction, true)
        correct = true == prediction.round()
        accuracy = correct.numpy().mean()
        em_ratio = np.mean(correct.numpy().all(1)).tolist()
        h_loss = hamming_loss(true.cpu(), prediction.round())
        self.log_dict(
            {
                "val/loss": loss,
                "val/accuracy": accuracy,
                "val/exact-match-ratio": em_ratio,
                "val/hamming-loss": h_loss,
            }
        )
