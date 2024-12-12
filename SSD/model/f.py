from torch import nn
import torch.nn.init as init
import torch
import sys
sys.path.append("../")
from utils import find_IoU

if not torch.cuda.is_available():
    print("CUDA unavailable")
    exit()
device = torch.device("cuda")

def xy2cxcy(boxes):
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]], 1)
        
def cxcy2xy(boxes):
    return torch.cat([boxes[:, :2] - (boxes[:, 2:] / 2), boxes[:, :2] + (boxes[:, 2:] / 2)], 1)

def encode_boxes(boxes,  default_boxes):
    epsilon = 1e-6
    return torch.cat([(boxes[:, :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] / 10 + epsilon), torch.log(boxes[:, 2:] / default_boxes[:, 2:] + epsilon) *5],1)

def decode_boxes(offsets, default_boxes):
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] / 10 + default_boxes[:, :2], torch.exp(offsets[:, 2:] / 5) * default_boxes[:, 2:]], 1)

class MultiBoxLoss(nn.Module):

    def __init__(self, default_boxes, threshold = 0.5, neg_pos= 3, alpha = 0.5):
        super(MultiBoxLoss, self).__init__()
        self.default_boxes = default_boxes
        self.threshold = threshold
        self.neg_pos = neg_pos
        self.alpha = alpha
        
    def forward(self, locs_pred, cls_pred, boxes, labels):

        batch_size = locs_pred.size(0)    
        n_def_boxes = self.default_boxes.size(0)    
        num_classes = cls_pred.size(2)    
        
        # location loss
        t_locs = torch.zeros((batch_size, n_def_boxes, 4), dtype= torch.float).to(device) 
        t_classes = torch.zeros((batch_size, n_def_boxes), dtype= torch.long).to(device)    
        
        def_boxes_xy = cxcy2xy(self.default_boxes)
        for i in range(batch_size):
            n_objects= boxes[i].size(0)
            overlap = find_IoU(boxes[i], def_boxes_xy)     
            
            overlap_defbox, object_defbox = overlap.max(dim= 0)  # overlap & object for each default box
            _, defbox_object = overlap.max(dim= 1)  # defbox for object
            object_defbox[defbox_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_defbox[defbox_object] = 1.

            label_defbox = labels[i][object_defbox]   
            label_defbox[overlap_defbox < self.threshold] = 0    

            t_classes[i] = label_defbox
            t_locs[i] = encode_boxes(xy2cxcy(boxes[i][object_defbox]), self.default_boxes)   

        pos_defboxes = t_classes != 0  
        
        loc_loss = nn.SmoothL1Loss(locs_pred[pos_defboxes], t_locs[pos_defboxes])

        n_positive = pos_defboxes.sum(dim= 1)
        n_hard_negatives = self.neg_pos * n_positive
        
        # confidence loss
        all_conf_loss = nn.CrossEntropyLoss(cls_pred.view(-1, num_classes), t_classes.view(-1), reduce=False).view(batch_size, n_def_boxes)
        
        pos_conf_loss = all_conf_loss[pos_defboxes]
        neg_conf_loss = all_conf_loss.clone()    
        neg_conf_loss[pos_defboxes] = 0.
        neg_conf_loss, _ = neg_conf_loss.sort(dim= 1, descending= True)
        
        hardness_ranks = torch.LongTensor(range(n_def_boxes)).unsqueeze(0).expand_as(neg_conf_loss).to(device) 
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  
        conf_hard_neg_loss = neg_conf_loss[hard_negatives]
        
        conf_loss = (conf_hard_neg_loss.sum() + pos_conf_loss.sum()) / n_positive.sum().float()
        
        return self.alpha * loc_loss + conf_loss

class Metrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class L2Norm(nn.Module):
    def __init__(self, channels, scale):
        super(L2Norm, self).__init__()
        self.channels = channels
        self.scale = scale
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, channels, 1, 1)) 
        init.constant_(self.rescale_factors, self.scale) # reset params
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = x / norm
        out = x * self.rescale_factors
        return out