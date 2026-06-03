

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os

from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform

        self.files = list(os.listdir(root))
        self.files = [
            p for p in self.files
            if p.endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        try:
            image_path = os.path.join(self.root, self.files[index])

            image = Image.open(image_path).convert("RGB")

            image = self.transform(image)

            return image

        except Exception as e:

            print(f"Error loading {image_path}: {e}")

            return self.__getitem__((index + 1) % len(self.files))
def get_transforms(size, crop, final_size):

    all_transformations = []

    if size > 0:
        all_transformations.append(
            transforms.Resize((size, size))
        )

    if crop:
        all_transformations.append(
            transforms.RandomCrop(final_size)
        )

    all_transformations.append(transforms.ToTensor())

    return transforms.Compose(all_transformations)
def adaptive_instance_normalization(content_features,style_features):

    size = content_features.size()

    content_mean, content_std = calculate_mean_std(
        content_features
    )

    style_mean, style_std = calculate_mean_std(
        style_features
    )

    content_normalized = (
        content_features - content_mean.expand(size)
    ) / content_std.expand(size)

    t = (
        style_std.expand(size) * content_normalized
    ) + style_mean.expand(size)

    return t

    
def calculate_mean_std(feature, eps=1e-5):

    size = feature.size()
    assert len(size) == 4

    batch_size, channels = size[:2]

    feature_view = feature.view(batch_size, channels, -1)

    feature_mean = feature_view.mean(dim=2).view(
        batch_size, channels, 1, 1
    )

    feature_var = feature_view.var(
        dim=2,
        unbiased=False
    ) + eps

    feature_std = feature_var.sqrt().view(
        batch_size, channels, 1, 1
    )

    return feature_mean, feature_std

    
