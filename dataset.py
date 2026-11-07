
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects",
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def standardize_image(tensor):
    """Remove each image's overall brightness and contrast."""
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)


class CompositeDataset(Dataset):
    def __init__(
        self,
        split,
        train=False,
        csv_path="data/splits.csv",
        neutral=False,
    ):
        with open(csv_path, newline="") as fh:
            self.rows = [
                r for r in csv.DictReader(fh)
                if r["split"] == split
            ]

        if not self.rows:
            raise ValueError(
                f"No images found for split '{split}'"
            )

        steps = [transforms.Resize((224, 224))]

        if neutral:
            steps.append(
                transforms.Grayscale(num_output_channels=3)
            )

        if train:
            steps += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        steps.append(transforms.ToTensor())

        if neutral:
            steps.append(transforms.Lambda(standardize_image))
        else:
            steps.append(
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            )

        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as im:
            image = im.convert("RGB")

        return (
            self.transform(image),
            CLASS_TO_INDEX[row["label"]],
        )


def make_loader(
    split,
    batch_size=32,
    allow_test=False,
    neutral=False,
):
    if split == "test" and not allow_test:
        raise ValueError(
            "The test split is locked. Use it once, at the very end."
        )

    dataset = CompositeDataset(
        split,
        train=(split == "train"),
        neutral=neutral,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
    )
