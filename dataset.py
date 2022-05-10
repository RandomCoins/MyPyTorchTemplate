from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class ImageSet(Dataset):
    def __init__(self, path, mode='train'):
        image_path = [line.rstrip('\n') for line in open(path)]
        self.mode = mode
        self.images = []
        self.labels = []
        for img in image_path:
            self.images.append(img.split()[0])
            self.labels.append(np.float32(img.split()[1]))

    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        image_name = self.images[item]
        score = self.labels[item]
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            image = transform(image)
            return image, np.array(5 - score, dtype=np.float32)
        else:
            image = transforms.ToTensor()(image)
            return image, score, image_name

    def __len__(self):
        return len(self.images)