import xml.etree.ElementTree as ET
import os, json

train_img = []
train_box = []
train_path = os.path.join("./dataset_s/train")
for f in os.listdir(train_path):
    if f.endswith('.jpg'):
        train_img.append(train_path + '/' + f)
        #train_img.append(f)
    else: 
        root = ET.parse(train_path + '/' + f).getroot()
        xmin = root.find('object/bndbox/xmin').text
        ymin = root.find('object/bndbox/ymin').text
        xmax = root.find('object/bndbox/xmax').text
        ymax = root.find('object/bndbox/ymax').text
        train_box.append({'boxes':[[int(xmin), int(ymin), int(xmax), int(ymax)]], 'labels':[1], 'difficulties':[0]})
        #train_box.append({'boxes':[[int(xmin), int(ymin), int(xmax), int(ymax)]], 'labels':[1]})


with open(os.path.join("./JSON", "train_img.json"), "w") as j:
    json.dump(train_img, j)
with open(os.path.join("./JSON", "train_obj.json"), "w") as j:
    json.dump(train_box, j)

test_img = []
test_box = []
test_path = os.path.join("./dataset_s/test")
for f in os.listdir(test_path):
    if f.endswith('.jpg'):
        test_img.append(test_path + '/' + f)
        #test_img.append(f)
    else: 
        root = ET.parse(test_path + '/' + f).getroot()
        xmin = root.find('object/bndbox/xmin').text
        ymin = root.find('object/bndbox/ymin').text
        xmax = root.find('object/bndbox/xmax').text
        ymax = root.find('object/bndbox/ymax').text
        test_box.append({'boxes':[[int(xmin), int(ymin), int(xmax), int(ymax)]], 'labels':[1], 'difficulties':[0]})
        #test_box.append({'boxes':[[int(xmin), int(ymin), int(xmax), int(ymax)]], 'labels':[1]})

with open(os.path.join("./JSON", "test_img.json"), "w") as j:
    json.dump(test_img, j)
with open(os.path.join("./JSON", "test_obj.json"), "w") as j:
    json.dump(test_box, j)



