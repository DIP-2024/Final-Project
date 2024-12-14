'''
Given an image and its corresponding labels, this script pixelates the objects in the image.
'''
from import_files import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


draw_boxes = False


image_path = "example/example.jpg"
image = Image.open(image_path).convert("RGB")
img_w, img_h = image.size


labels_path = "example/example.txt"
boxes, labels = [], []
with open(labels_path, 'r') as f:
    for line in f:
        label, box = YOLOv8Format(img_w, img_h, line)
        labels.append(label)
        boxes.append(box)


draw_image = image.copy()
for box in boxes:
    draw_image = pixelate_box(draw_image, box, pixel_size=5)


# Draw the bounding boxes (optional, for visualization)
if draw_boxes:
    draw = ImageDraw.Draw(draw_image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)


# Display the result
plt.imshow(draw_image)
plt.axis("off")
plt.show()


# Save the result
draw_image.save("example/example_pixelated.jpg")


print('Done!')