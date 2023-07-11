from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration

CONDITIONAL_TEXT = "An album cover"

dataset300 = load_dataset("mnne/covers300")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda")


def describe_sample(sample):
    if "caption" in sample:
        print("jj")
    image = sample["image"]
    inputs = processor(


        
        images=image,
        text=[CONDITIONAL_TEXT for _ in range(len(image))],
        return_tensors="pt",
    ).to("cuda")
    caption = model.generate(**inputs, max_new_tokens=50)
    sample["caption"] = [processor.decode(c, skip_special_tokens=True) for c in caption]
    return sample


dataset300 = dataset300.map(
    describe_sample,
    batched=True,
    batch_size=32,
    load_from_cache_file=True,
)
dataset300.push_to_hub("mnne/covers300", private=True)

dataset64 = load_dataset("mnne/covers64")


# for subset, name in dataset64.items():
#     for sample in subset:
