import torch
from torchvision.transforms import functional as F
from PIL import Image
import os


def YOLOv8Format(img_w, img_h, line):
    '''
    Convert YOLO format to absolute box coordinates
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    '''
    # <class_id> <x_center> <y_center> <width> <height>
    parts = line.strip().split()

    label = int(parts[0])

    x, y, w, h = map(float, parts[1:])
    # Convert YOLO format to absolute box coordinates
    xmin = (x - w / 2) * img_w
    ymin = (y - h / 2) * img_h
    xmax = (x + w / 2) * img_w
    ymax = (y + h / 2) * img_h
    box = [xmin, ymin, xmax, ymax]

    return label, box


class YOLOv8Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, num_labels, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(images_dir))
        self.num_labels = num_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Convert the image to a tensor
        img = F.to_tensor(img)

        # Load annotation (assumes YOLO format for simplicity)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        boxes, labels = [], []
        with open(ann_path, 'r') as f:
            for line in f:
                label, box = YOLOv8Format(img.shape[2], img.shape[1], line)
                labels.append(label)
                boxes.append(box)
                # parts = line.strip().split()

                # labels.append(int(parts[0]))

                # x, y, w, h = map(float, parts[1:])
                # # Convert YOLO format to absolute box coordinates
                # img_w, img_h = img.shape[2], img.shape[1]
                # xmin = (x - w / 2) * img_w
                # ymin = (y - h / 2) * img_h
                # xmax = (x + w / 2) * img_w
                # ymax = (y + h / 2) * img_h
                # boxes.append([xmin, ymin, xmax, ymax])

        # Convert to PyTorch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target