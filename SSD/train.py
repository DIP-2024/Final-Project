import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data import VOCDataset
from utils import combine
import sys
sys.path.append("./model/")
from model.SSD300 import SSD300
from model.f import MultiBoxLoss, Metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if not torch.cuda.is_available():
    print("CUDA unavailable")
    exit()
device = torch.device("cuda")

data_folder = "./JSON"
num_classes = 2
###########################################################################
checkpoint = "./ssd300_b32.pth.tar"
batch_size = 32     
lr = 0.001 
decay_lr_to = 0.1  
epochs = 30
lr_decay = [40]  # @epoch
###########################################################################
workers = 4
momentum = 0.9  
weight_decay = 0.0001
cudnn.benchmark = True
###########################################################################

def main():
    global start_epoch, epoch, checkpoint
    
    if checkpoint is None:
        start_epoch= 0
        model = SSD300(num_classes)
        biases = list()
        not_biases = list()
        for pname, param in model.named_parameters():
            if param.requires_grad:
                if pname.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = optim.SGD(params= [{'params': biases,'lr': 2* lr, 'weight_decay': 0}, {'params': not_biases, 'lr':lr, 'weight_decay': weight_decay}], momentum = momentum)
    
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1   
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(model.default_boxes).to(device)
    
    train_dataset = VOCDataset(data_folder, split= "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle= True, collate_fn= combine, num_workers = workers, pin_memory = True)
    
    for epoch in range(start_epoch, epochs):
        # decay lr
        if epoch in lr_decay:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_lr_to
            print("New LR: {}".format, (optimizer.param_groups[1]['lr']))
        
        train(train_loader = train_loader, model = model, criterion= criterion, optimizer = optimizer, epoch = epoch)

        #save checkpoint
        state = {'epoch': epoch, "model": model, "optimizer": optimizer}
        filename = "ssd300_b32.pth.tar"
        torch.save(state, filename)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    losses = Metrics()
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        
        images = images.to(device)  
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        locs_pred, cls_pred = model(images) 
        loss = criterion(locs_pred, cls_pred, boxes, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        
        if i == len(train_loader)-1:
            print('Epoch {0}: Loss {loss.val:.4f} (Average Loss: {loss.avg:.4f})'.format(epoch, loss=losses))

    del locs_pred, cls_pred, images, boxes, labels

if __name__ == '__main__':
    main()