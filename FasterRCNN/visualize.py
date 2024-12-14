'''
Given an image and a trained model, this script will visualize the predictions of the model on the image.
'''
from import_files import *
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import time


time_start = time.time()


# Check if GPU is available
device = torch.device('cuda' if check_gpu() else 'cpu')


model_path = 'fine_tuned/faster_rcnn.pth'
model = torch.load(model_path)


image_path = "example/example.jpg"
image = Image.open(image_path).convert("RGB")


# Preprocess the image for the model
image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension


# Put the model in evaluation mode
model.eval()
with torch.no_grad():
    outputs = model(image_tensor)


# Extract predictions
predictions = outputs[0]
boxes = predictions['boxes'].cpu().numpy()
scores = predictions['scores'].cpu().numpy()
labels = predictions['labels'].cpu().numpy()


confidence_threshold = 0.5
filtered_indices = scores > confidence_threshold
filtered_boxes = boxes[filtered_indices]
filtered_labels = labels[filtered_indices]


draw_image = image.copy()
for box in filtered_boxes:
    draw_image = pixelate_box(draw_image, box, pixel_size=10)


# Draw the bounding boxes (optional, for visualization)
draw = ImageDraw.Draw(draw_image)
for box in filtered_boxes:
    draw.rectangle(box, outline="red", width=3)


# Display the result
plt.imshow(draw_image)
plt.axis("off")
plt.show()


print('Total runnning time:', format_time(time.time() - time_start))
print('Done!')