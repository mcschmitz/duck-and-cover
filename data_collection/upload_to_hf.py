from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("imagefolder", data_dir="./data/covers64")
    dataset.push_to_hub("mnne/covers64", private=True)

    dataset = load_dataset("imagefolder", data_dir="./data/covers300")
    dataset.push_to_hub("mnne/covers300", private=True)
