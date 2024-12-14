'''
This script performs object detection on a video using a pre-trained Faster R-CNN model.
'''
from import_files import *
import cv2
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Check if GPU is available
device = torch.device('cuda' if check_gpu() else 'cpu')


# Initialize the model
model_path = 'fine_tuned/faster_rcnn.pth'
model = torch.load(model_path)
model.to(device)


# Load video
video_path = "example/example.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)


# Get video properties for saving output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))


# Detection threshold
confidence_threshold = 0.8


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess for the model
    image_tensor = F.to_tensor(pil_image).unsqueeze(0).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract predictions
    predictions = outputs[0]
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    # Filter predictions
    filtered_indices = scores > confidence_threshold
    filtered_boxes = boxes[filtered_indices]
    filtered_scores = scores[filtered_indices]
    filtered_labels = [str(label) for label in labels[filtered_indices]]

    # Draw predictions on the frame
    for box in filtered_boxes:
        pil_image = pixelate_box(pil_image, box, pixel_size=5)
    annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Write and display the frame
    out.write(annotated_frame)
    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
