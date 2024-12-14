import os, cv2, shutil
import numpy as np
import xml.etree.cElementTree as ET

# {"smoke":"0"}

data_root = "./dataset_s/test"
imgs_path = os.listdir(data_root)
imgs_path = [os.path.join(data_root, i) for i in imgs_path]

for i in range(len(imgs_path)):
    if (imgs_path[i]).endswith('.xml'):
        continue

    img = cv2.imread(imgs_path[i])
    save_img = os.path.join('./savedata_test/', "test{}.jpg".format(i))
    save_txt = os.path.join('./savedata_test/', "test{}.txt".format(i))
    with open(save_txt, "w") as f:
        h, w = img.shape[:2]
        xml_path = imgs_path[i][:-4] + ".xml"
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall('object'):
            xmin = obj.find('bndbox').find('xmin').text
            ymin = obj.find('bndbox').find('ymin').text
            xmax = obj.find('bndbox').find('xmax').text
            ymax = obj.find('bndbox').find('ymax').text
            boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        if len(boxes) == 0:
            continue
        for idx, b in enumerate(boxes):
            annotation = np.zeros((1, 4))
            annotation[0, 0] = (b[0] + (b[2] -b[0]) / 2) / w  # cx
            annotation[0, 1] = (b[1] + (b[3]-b[1]) / 2) / h  # cy
            annotation[0, 2] = (b[2] -b[0]) / w  
            annotation[0, 3] = (b[2] -b[0]) / h  
            str_label = "0"

            for i in range(len(annotation[0])):
                str_label = str_label + " " + str(annotation[0][i])

            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            f.write(str_label)
    
    cv2.imwrite(save_img, img)

###################################################################################

label_path = "./savedata_testlabels"
if not os.path.exists(label_path):
    os.mkdir(label_path)
img_path = "./savedata_test/images"
if not os.path.exists(img_path):
    os.mkdir(img_path)

old_path =  "./savedata_test"
for i in os.listdir(old_path):
    if i.endswith(".txt"):
        shutil.move(os.path.join(old_path,i),os.path.join(label_path,i))
    else: 
        shutil.move(os.path.join(old_path,i),os.path.join(img_path,i))
