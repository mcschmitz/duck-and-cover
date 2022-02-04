import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BatchEncoding, BertTokenizer

from utils import logger


class GenreDataLoader:
    def __init__(
        self,
        meta_data_path: str,
        tokenizer_path: str,
        batch_size: int = 32,
    ):
        """
        Genre loader that takes a path to the meta data frame, loads it and
        returns the genre information.

        Args:
            meta_data_path: Path to the json file that contains information
                about the training data.
            tokenizer_path: Path to a directory that holds the tokenizer's
                vocab file. If not exists, will train a tokenizer.
            batch_size: Size of one Batch
        """
        self.meta_df = pd.read_json(
            meta_data_path, orient="records", lines=True
        )
        self.meta_df = self.meta_df.dropna(
            subset=["file_path_64", "file_path_300"]
        )
        self.meta_df = self.meta_df.dropna(subset=["artist_genre"])
        self.tokenizer = self.setup_tokenizer(tokenizer_path)
        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit(self.meta_df["artist_genre"].to_list())
        train_set = self.meta_df.sample(frac=0.7)
        data = self.meta_df[~self.meta_df.index.isin(train_set.index)]
        self.val_set = data.sample(frac=2 / 3)
        self.test_set = data[~data.index.isin(self.val_set.index)]
        self.train_generator = GenreTrainGenerator(
            data=train_set,
            batch_size=batch_size,
            tokenizer=self.tokenizer,
            binarizer=self.label_binarizer,
        )
        self.val_generator = GenreDatasetGenerator(
            data=self.val_set,
            batch_size=256,
            tokenizer=self.tokenizer,
            binarizer=self.label_binarizer,
        )

    def setup_tokenizer(self, path: str):
        """
        Trains a Bert Tokenizer on the comma separated list of genre strings or
        loads a tokenizer from a given path.

        Args:
            path: Tokenizer directory that holds the vocab file
        """
        if os.path.exists(f"{path}/vocab.txt"):
            return BertTokenizer(vocab_file=f"{path}/vocab.txt")
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False,
        )
        tokenizer.train_from_iterator(
            GenreDataLoader._batch_iterator(self.meta_df, 50000),
        )
        Path(path).mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(path)
        return BertTokenizer(vocab_file=f"{path}/vocab.txt")

    def get_number_of_classes(self) -> int:
        """
        Returns the total number of genres.
        """
        return len(self.label_binarizer.classes_)

    @classmethod
    def _batch_iterator(cls, df, batch_size=1000):
        for i in range(0, len(df), batch_size):
            subframe = df[i : i + batch_size]
            genre_lists = subframe["artist_genre"].tolist()
            genre_lists = [sorted(gl) for gl in genre_lists]
            yield [" ".join(gl) for gl in genre_lists]


class GenreTrainGenerator:
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        tokenizer: BertTokenizer,
        binarizer: MultiLabelBinarizer,
    ):
        """
        Trainset generator for the  gerne data.

        Args:
            data: Dataset to be used for training
            batch_size: training batch size
            tokenizer: BertWordPieceTokenizer that should be used to tokenize
                the concatenated genre list.
            binarizer: MultiLabelBinarizer to binarize the actual gerne lables
        """
        self._iterator_i = 0
        self.batch_size = batch_size
        self.data = data.reset_index(drop=True)
        self.n_samples = len(self.data)
        self.tokenizer = tokenizer
        self.label_binarizer = binarizer

    def __iter__(self):
        yield from (self[batch_id] for batch_id in range(len(self)))

    def __len__(self):
        return self.n_samples // self.batch_size

    def __getitem__(self, item) -> Tuple[BatchEncoding, torch.Tensor]:
        genre_strings = []
        labels_list = []
        batch_idx = self._get_batch_idx()
        for b_idx in batch_idx:
            labels_list.append(self.data["artist_genre"][b_idx])
            genre_strings.append(" ".join(self.data["artist_genre"][b_idx]))
        self._iterator_i = batch_idx[-1]
        labels = self.label_binarizer.transform(labels_list)
        return (
            self.tokenizer(
                genre_strings,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ),
            torch.Tensor(labels),
        )

    def _get_batch_idx(self):
        positions = np.arange(
            self._iterator_i, self._iterator_i + self.batch_size
        )
        batch_idx = [
            i if i < self.n_samples else i - self.n_samples for i in positions
        ]
        if 0 in batch_idx:
            logger.info("Data Generator exceeded. Will shuffle input data.")
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        return batch_idx


class GenreDatasetGenerator:
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        tokenizer: BertTokenizer,
        binarizer: MultiLabelBinarizer,
    ):
        """
        Trainset generator for the  gerne data.

        Args:
            data: Dataset to be used for training
            batch_size: training batch size
            tokenizer: BertWordPieceTokenizer that should be used to tokenize
                the concatenated genre list.
            binarizer: MultiLabelBinarizer to binarize the actual gerne lables
        """
        self.batch_size = batch_size
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_binarizer = binarizer
        genre_strings = []
        labels_list = []
        for _idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            labels_list.append(row["artist_genre"])
            genre_strings.append(" ".join(row["artist_genre"]))
        self.labels = torch.Tensor(self.label_binarizer.transform(labels_list))
        self.data = self.tokenizer(
            genre_strings,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        self.data = {
            k: torch.split(v, self.batch_size) for k, v in self.data.items()
        }
        self.labels = torch.split(self.labels, self.batch_size)

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        for i, labels in enumerate(self.labels):
            batch = {k: v[i] for k, v in self.data.items()}
            yield (batch, labels)
