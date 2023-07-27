from datasets import load_dataset, Dataset
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration

CONDITIONAL_TEXT = "An album cover showing"


def describe_sample(
    samples: dict,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
) -> dict:
    if "caption" not in samples:
        samples["caption"] = [None for _ in range(len(samples["image"]))]
    captions = samples["caption"]
    idx = [i for i, c in enumerate(captions) if c is None]
    if idx:
        images = [samples["image"][i] for i in idx]
        inputs = processor(
            images=images,
            text=[CONDITIONAL_TEXT for _ in range(len(images))],
            return_tensors="pt",
        ).to("cuda")
        captions = model.generate(**inputs, max_new_tokens=50)
        for i, c in zip(idx, captions):
            samples["caption"][i] = processor.tokenizer.decode(
                c, skip_special_tokens=True
            )
    return samples


def map_captions(samples: dict, mapping_df: pd.DataFrame) -> dict:
    samples["caption"] = [None for _ in range(len(samples["album_id"]))]
    for i, album_id in enumerate(samples["album_id"]):
        album_d = mapping_df[mapping_df["album_id"] == album_id]
        if len(album_d) > 0:
            samples["caption"][i] = album_d["caption"].values[0]
    return samples


if __name__ == "__main__":
    dataset64 = load_dataset("imagefolder", data_dir="./data/covers64")
    dataset300 = load_dataset("imagefolder", data_dir="./data/covers300")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cuda")

    dataset300 = dataset300.map(
        describe_sample,
        batched=True,
        batch_size=128,
        desc="Generating captions for Covers300",
        fn_kwargs={"processor": processor, "model": model},
    )
    dataset300.push_to_hub("mnne/covers300", private=True)

    dataset300_df = pd.DataFrame(
        {
            "album_id": dataset300["train"]["album_id"],
            "caption": dataset300["train"]["caption"],
        }
    )

    dataset64 = dataset64.map(
        map_captions,
        batched=True,
        batch_size=128,
        desc="Mapping captions from Covers300 to Covers64",
        fn_kwargs={"mapping_df": dataset300_df},
    )
    dataset64 = dataset64.map(
        describe_sample,
        batched=True,
        batch_size=128,
        desc="Generating captions for Covers64",
        fn_kwargs={"processor": processor, "model": model},
    )

    dataset64.push_to_hub("mnne/covers64", private=True)
