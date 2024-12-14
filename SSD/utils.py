import torch, warnings
warnings.filterwarnings("ignore", category=UserWarning)

if not torch.cuda.is_available():
    print("CUDA unavailable")
    exit()
device = torch.device("cuda") 
 
# mAP, SSD, f
def find_IoU(boxes1, boxes2):
    #intersect
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy =  torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2), boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2), boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy , min=0)  # (n1, n2, 2)
    intersect = inter[:, :, 0] * inter[:, :, 1]  #(n1, n2)

    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(intersect)  #(n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(intersect)  #(n1, n2)
    union = (area_boxes1 + area_boxes2 - intersect)
    return intersect / union

# train.py, eval.py
def combine(batch):
    images = []
    boxes = []
    labels = []
    difficulties = []
    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])
    images = torch.stack(images, dim= 0)
    return images, boxes, labels, difficulties

# eval.py
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):

    true_images = []
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  
    true_boxes = torch.cat(true_boxes, dim=0) 
    true_labels = torch.cat(true_labels, dim=0)  
    true_difficulties = torch.cat(true_difficulties, dim=0)  

    det_images = []
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  
    det_boxes = torch.cat(det_boxes, dim=0)  
    det_labels = torch.cat(det_labels, dim=0)  
    det_scores = torch.cat(det_scores, dim=0)   
 
    true_det_boxes = torch.zeros((true_difficulties.size(0)), dtype=torch.uint8).to(device)  
    n_detections = det_boxes.size(0)

    det_scores, sort_ind = torch.sort(det_scores, dim=0, descending=True) 
    det_images = det_images[sort_ind]  
    det_boxes = det_boxes[sort_ind]  

    # true, false positives
    true_pos = torch.zeros((n_detections), dtype=torch.float).to(device)  
    false_pos = torch.zeros((n_detections), dtype=torch.float).to(device)  
    for d in range(n_detections):
        this_det_box = det_boxes[d].unsqueeze(0)  # (1, 4)
        this_image = det_images[d]  

        object_boxes = true_boxes[true_images == this_image]  
        if object_boxes.size(0) == 0:
            false_pos[d] = 1
            continue

        overlaps = find_IoU(this_det_box, object_boxes)  
        max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  

        original_ind = torch.LongTensor(range(true_boxes.size(0))).to(device)[true_images == this_image][ind]
        
        if max_overlap.item() > 0.5:
            if true_det_boxes[original_ind] == 0:
                true_pos[d] = 1
                true_det_boxes[original_ind] = 1  
            else:
                false_pos[d] = 1
        else:
            false_pos[d] = 1

    c_true_pos = torch.cumsum(true_pos, dim=0)  
    c_false_pos = torch.cumsum(false_pos, dim=0)  
    c_precision = c_true_pos / (c_true_pos + c_false_pos + 1e-10)  
    c_recall = c_true_pos / len(true_labels)  

    recall_thr = torch.arange(start=0, end=1.1, step=.1).tolist()
    precisions = torch.zeros((len(recall_thr)), dtype=torch.float).to(device)  
    for i, t in enumerate(recall_thr):
        recalls_above_t = c_recall >= t
        if recalls_above_t.any():
            precisions[i] = c_precision[recalls_above_t].max()
        else:
            precisions[i] = 0.
 
    mAP = precisions.mean().item()

    return mAP