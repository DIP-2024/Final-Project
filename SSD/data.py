import torch, os, json
from torch.utils.data import Dataset
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, DataFolder, split):
        self.split = split
        self.DataFolder = DataFolder
        
        with open(os.path.join(DataFolder, self.split+ '_img.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(DataFolder, self.split+ '_obj.json'), 'r') as j:
            self.objects = json.load(j)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        
        image = Image.open(self.images[i], mode= "r")
        image = image.convert("RGB")
        
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels']) 
        difficulties = torch.ByteTensor(objects['difficulties'])
        
        return image, boxes, labels, difficulties
        
            
        