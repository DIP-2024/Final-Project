'''
Train a Faster R-CNN model on the smoking detection dataset
'''
from import_files import *
import yaml
import time
from torchvision.transforms import transforms
from torch.utils.data import random_split
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchinfo import summary
import torch.optim as optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os


time_start = time.time()


#=== Hyperparameters ===#
batch_size = 8
val_size = 0.2
num_epochs = 50

# Save the model
save_model = False
save_path = 'fine_tuned'
model_name = 'faster_rcnn.pth'
#=======================#



# Check if GPU is available
device = torch.device('cuda' if check_gpu() else 'cpu')



#===============================
# Load the dataset
#===============================
# data_path = "Smooking-Detection-4/"
data_path = 'smoking-detection-1/'
# Load the .yml file
with open(os.path.join(data_path, 'data.yaml'), 'r') as file:
    config = yaml.safe_load(file)

for k, v in config.items():
    print(k + ':', v)

# Access paths and class names
train_dir = config['train']
val_dir = config['val']
test_dir = config['test']
class_names = config['names']
num_classes = config['nc'] + 1  # Including the background class

# Paths to your dataset
train_images_dir = os.path.join(data_path, 'train', 'images')
train_annotations_dir = os.path.join(data_path, 'train', 'labels')
test_images_dir = os.path.join(data_path, 'test', 'images')
test_annotations_dir = os.path.join(data_path, 'test', 'labels')

# Create datasets
train_dataset = YOLOv8Dataset(train_images_dir, train_annotations_dir, num_classes)
train_dataset, val_dataset = random_split(train_dataset, [1 - val_size, val_size])
test_dataset = YOLOv8Dataset(test_images_dir, test_annotations_dir, num_classes)

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(val_dataset)}")
print(f"Number of validation examples: {len(test_dataset)}")



#===============================
# Create data loaders
#===============================
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Check the shape of the first batch
train_features, train_labels = next(iter(train_loader))
print(f"Feature picture shape: {train_features[0].shape}")
print(f"Labels labels shape: {train_labels[0]['labels'].shape}")
print(f"Labels boxes shape: {train_labels[0]['boxes'].shape}")
# print(f"Labels labels shape: {train_labels[0]['labels'].dtype}")
# exit()



#===============================
# Load the pre-trained model
#===============================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the head for your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

summary(model)


#===============================
# Initialize optimizer and learning rate scheduler
#===============================
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Initialize metric for mAP
map_metric = MeanAveragePrecision()



#===============================
# Training loop
#===============================
model.to(device)
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    time_epoch_start = time.time()

    model.train()
    total_loss = 0  # To track epoch loss

    #=== Training phase ===#
    total_size = len(train_loader)
    for i, (images, targets) in enumerate(train_loader):
        print(f'\r{i+1}/{total_size}', end='')

        # Move images and targets to the device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if any(len(t["boxes"]) == 0 for t in targets):
            continue

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += losses.item()
    
    print(f' training time: {format_time(time.time() - time_epoch_start)}')

    lr_scheduler.step()

    #=== Validation phase ===#
    model.eval()
    map_metric.reset()  # Reset metrics for new epoch
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = [img.to(device) for img in val_images]
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            predictions = model(val_images)
            map_metric.update(predictions, val_targets)

    # Compute mAP and log results
    metrics = map_metric.compute()
    print(
        f"Validation:"
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss / len(train_loader):.4f}, "
        f"mAP: {metrics['map']:.4f}, "
        f"mAP@50: {metrics['map_50']:.4f}, "
        f"mAP@75: {metrics['map_75']:.4f}"
    )

    print(f"Epoch time: {format_time(time.time() - time_epoch_start)}")


print(f"Training time: {format_time(time.time() - time_start)}")



#===============================
# Test the model
#===============================
model.eval()
map_metric.reset()
with torch.no_grad():
    for test_images, test_targets in test_loader:
        test_images = [img.to(device) for img in test_images]
        test_targets = [{k: v.to(device) for k, v in t.items()} for t in test_targets]

        predictions = model(test_images)
        map_metric.update(predictions, test_targets)

metrics = map_metric.compute()
print(
    f"Test:"
    f"mAP: {metrics['map']:.4f}, "
    f"mAP@50: {metrics['map_50']:.4f}, "
    f"mAP@75: {metrics['map_75']:.4f}"
)



#===============================
# Save the model
#===============================
if save_model:
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))



print('Total runnning time:', format_time(time.time() - time_start))
print("Done!")