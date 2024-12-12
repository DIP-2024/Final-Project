import sys
sys.path.append("./model/")
from utils import *
from data import VOCDataset
import tqdm, torch, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if not torch.cuda.is_available():
    print("CUDA unavailable")
    exit()
device = torch.device("cuda")

trained_model = torch.load("ssd300_b32.pth.tar")
test_dataset = VOCDataset("./JSON", split= "test")
batch_size = 32
###########################################################################
workers = 4
model = trained_model["model"]
model = model.to(device)
# for NMS
min_score = 0.1
max_overlap = 0.45
top_k = 200
###########################################################################

model.eval()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, shuffle= False, collate_fn = combine, num_workers= workers, pin_memory= True)

def evaluate(model, test_loader):
    model.eval()
    
    detect_boxes = []
    detect_labels = []
    detect_scores = []
    t_boxes = []
    t_labels = []
    t_difficulties = []
    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc= "Evaluating")):
            images = images.to(device)
            
            locs_pred, cls_pred = model(images)
            detect_boxes_batch, detect_labels_batch, detect_score_batch = model.detect(locs_pred, cls_pred, min_score, max_overlap, top_k)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]
            
            detect_boxes.extend(detect_boxes_batch)
            detect_labels.extend(detect_labels_batch)
            detect_scores.extend(detect_score_batch)
            t_boxes.extend(boxes)
            t_labels.extend(labels)
            t_difficulties.extend(difficulties)
        
        mAP = calculate_mAP(detect_boxes, detect_labels, detect_scores, t_boxes, t_labels, t_difficulties)
        
    print("Mean Average Precision (mAP): {}".format(mAP))

if __name__ == '__main__':
    evaluate(model, test_loader)
    
