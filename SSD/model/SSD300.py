from torch import nn
import torch.nn.functional as F
import torch, math
import sys
sys.path.append("./")
from vgg import *
from f import *
from boxes import *
from utils import find_IoU

if not torch.cuda.is_available():
    print("CUDA unavailable")
    exit()
device = torch.device("cuda")

class SSD300(nn.Module):

    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        
        self.num_classes = num_classes
        self.base_model = VGG16BaseNet(pretrained= True)
        self.aux_net = AuxiliaryNet()
        self.pred_net = PredictionNet(num_classes)
        self.L2Norm = L2Norm(channels= 512, scale= 20)
        self.default_boxes = self.create_default_boxes()

    def forward(self, image):
        conv4_3_out, conv7_out = self.base_model(image)   
        conv4_3_out = self.L2Norm(conv4_3_out)    
        conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = self.aux_net(conv7_out) 
        
        locs_pred, cls_pred = self.pred_net(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)    
        return locs_pred, cls_pred
        
    def create_default_boxes(self):
        fmap_wh = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5, "conv10_2": 3, "conv11_2": 1}
        scales = {"conv4_3": 0.1, "conv7": 0.2, "conv8_2": 0.375, "conv9_2": 0.55, "conv10_2": 0.725, "conv11_2": 0.9}
        aspect_ratios= {"conv4_3": [1., 2., 0.5], "conv7": [1., 2., 3., 0.5, 0.3333], "conv8_2": [1., 2., 3., 0.5, 0.3333], "conv9_2": [1., 2., 3., 0.5, 0.3333], "conv10_2": [1., 2., 0.5], "conv11_2": [1., 2., 0.5]}
        fmaps = list(fmap_wh.keys())
        
        default_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_wh[fmap]):
                for j in range(fmap_wh[fmap]):
                    cx = (j + 0.5) / fmap_wh[fmap]
                    cy = (i + 0.5) / fmap_wh[fmap]
                    for ratio in aspect_ratios[fmap]:
                        default_boxes.append([cx, cy, scales[fmap]* math.sqrt(ratio), scales[fmap]/math.sqrt(ratio)]) 
                        if ratio == 1:
                            try:
                                add_scale = math.sqrt(scales[fmap]*scales[fmaps[k+1]])
                            except IndexError:
                                add_scale = 1.
                            default_boxes.append([cx, cy, add_scale, add_scale])
        
        default_boxes = torch.FloatTensor(default_boxes).to(device) 
        default_boxes.clamp_(0, 1)
        return default_boxes
    
    def detect(self, locs_pred, cls_pred, min_score, max_overlap, top_k):
        batch_size = locs_pred.size(0)  
        cls_pred = F.softmax(cls_pred, dim= 2)    
        
        all_boxes = []
        all_labels = []
        all_scores = []
        
        for i in range(batch_size):

            decoded_locs = cxcy2xy(decode_boxes(locs_pred[i], self.default_boxes)) 
            
            image_boxes = []
            image_labels = []
            image_scores = []

            scores = cls_pred[i][:, 1]    
            score_above = scores > min_score
            n_above = score_above.sum().item()
            
            if n_above > 0:
                scores = scores[score_above]    
                decoded_locs = decoded_locs[score_above] 
                scores, sort_id = scores.sort(dim= 0, descending= True)
                decoded_locs = decoded_locs[sort_id]
                
                overlap = find_IoU(decoded_locs, decoded_locs)
                
                #NMS
                suppress = torch.zeros((n_above), dtype=torch.bool).to(device)
                for box_id in range(n_above):
                    if suppress[box_id]:
                        continue
                    cond = overlap[box_id] > max_overlap
                    suppress |= cond
                    suppress[box_id] = 0
                
                image_boxes.append(decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [1]).to(device))
                image_scores.append(scores[1 - suppress])
            
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim= 0)    
            image_labels = torch.cat(image_labels, dim=0)  
            image_scores = torch.cat(image_scores, dim=0)  
            n_objects = image_scores.size(0)
            
            if n_objects > top_k:
                image_scores, sort_index = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  
                image_boxes = image_boxes[sort_index][:top_k]  
                image_labels = image_labels[sort_index][:top_k]  
            
            all_boxes.append(image_boxes)
            all_labels.append(image_labels)
            all_scores.append(image_scores)
            
        return all_boxes, all_labels, all_scores        