import torch, torchvision
from torch import nn
import torch.nn.functional as F

def decimate(tensor, m):
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d], dtype=torch.long))
    return tensor

class VGG16BaseNet(nn.Module):
    def __init__(self, pretrained= True):
        super(VGG16BaseNet, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding= 1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding= 1)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride= 2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size= (3, 3), padding= 1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size= (3, 3), padding= 1)
        self.pool2 = nn.MaxPool2d(kernel_size= (2, 2), stride= 2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size= (3, 3), padding= 1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size= (3, 3), padding= 1)
        self.pool3 = nn.MaxPool2d(kernel_size= (2, 2), stride= 2, ceil_mode= True)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size= (3, 3), padding= 1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size= (3, 3), padding= 1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size= (3, 3), padding= 1)
        self.pool4 = nn.MaxPool2d(kernel_size= (2, 2), stride= 2)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size= (3, 3), padding= 1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size= (3, 3), padding= 1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size= (3, 3), padding= 1)
        self.pool5 = nn.MaxPool2d(kernel_size= (3, 3), stride= 1, padding= 1)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size= (3, 3), padding= 6, dilation= 6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size= (1, 1))

        self.load_pretrained()
       

    def forward(self, image):
        x = image                      #(N, 3, 300, 300)

        #(N, 64, 300, 300)
        x = F.relu(self.conv1_1(x))    
        x = F.relu(self.conv1_2(x))    
        x = self.pool1(x)              
        #(N, 64, 150, 150)
        
        #(N, 128, 150, 150)
        x = F.relu(self.conv2_1(x))    
        x = F.relu(self.conv2_2(x))    
        x = self.pool2(x)              
        #(N, 128, 75, 75)
        
        #(N, 256, 75, 75)
        x = F.relu(self.conv3_1(x))    
        x = F.relu(self.conv3_2(x))    
        x = F.relu(self.conv3_3(x))    
        x = self.pool3(x)              
        #(N, 256, 38, 38)
        
        #(N, 512, 38, 38)
        x = F.relu(self.conv4_1(x))    
        x = F.relu(self.conv4_2(x))    
        x = F.relu(self.conv4_3(x))    
        conv4_3_out = x                
        x = self.pool4(x)              
        #(N, 512, 19, 19)
        
        #(N, 512, 19, 19)
        x = F.relu(self.conv5_1(x))    
        x = F.relu(self.conv5_2(x))    
        x = F.relu(self.conv5_3(x))    
        x = self.pool5(x)              
        x = F.relu(self.conv6(x))      
        #(N, 1024, 19, 19)
        
        conv7_out = F.relu(self.conv7(x))    #(N, 1024, 19, 19)
        
        return conv4_3_out, conv7_out
    
    def load_pretrained(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        
        pretrained_sd = torchvision.models.vgg16(pretrained= True).state_dict()
        
        for i, parameters in enumerate(param_names[:26]):
            state_dict[parameters] = pretrained_sd[list(pretrained_sd.keys())[i]]

        fc6_weight = pretrained_sd['classifier.0.weight'].view(4096, 512, 7, 7)
        fc6_bias = pretrained_sd['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(fc6_weight, m=[4, None, 3, 3])    #(1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(fc6_bias, m=[4])    
        
        fc7_weight = pretrained_sd['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias = pretrained_sd['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(fc7_bias, m=[4])
        
        self.load_state_dict(state_dict)
        print("Loaded base model")


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), padding= 0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size= (3, 3), padding= 1, stride= 2)
        
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size= (1, 1), padding= 0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 1, stride= 2)
        
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size= (1, 1), padding= 0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 0)
        
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size= (1, 1), padding= 0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 0)
        
        self.weights_init()
    
    def forward(self, conv7_out):
        x = conv7_out    #(N, 1024, 19, 19)
        x = F.relu(self.conv8_1(x))    #(N, 256, 19, 19)
        x = F.relu(self.conv8_2(x))    #(N, 512, 10, 10)
        conv8_2_out = x
        
        x = F.relu(self.conv9_1(x))    #(N, 128, 10, 10)
        x = F.relu(self.conv9_2(x))    #(N, 256, 5, 5)
        conv9_2_out = x
        
        x = F.relu(self.conv10_1(x))   #(N, 128, 5, 5)
        x = F.relu(self.conv10_2(x))   #(N, 256, 3, 3)
        conv10_2_out = x
        
        x = F.relu(self.conv11_1(x))              #(N, 128, 3, 3)
        conv11_2_out = F.relu(self.conv11_2(x))   #(N, 256, 1, 1)
        
        return conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out
    
    def weights_init(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)

class PredictionNet(nn.Module):
    def __init__(self, num_classes):
        super(PredictionNet, self).__init__()
        
        self.num_classes = num_classes
        
        num_boxes = {"conv4_3": 4, "conv7": 6, "conv8_2": 6, "conv9_2": 6, "conv10_2": 4, "conv11_2": 4}
        
        self.conv_4_3_loc = nn.Conv2d(512, num_boxes["conv4_3"] * 4, kernel_size= (3, 3), padding= 1)
        self.conv7_loc = nn.Conv2d(1024, num_boxes["conv7"] * 4, kernel_size= (3, 3), padding= 1)
        self.conv8_2_loc = nn.Conv2d(512, num_boxes["conv8_2"]*4, kernel_size= (3, 3), padding= 1)
        self.conv9_2_loc = nn.Conv2d(256, num_boxes["conv9_2"]*4, kernel_size= (3, 3), padding= 1)
        self.conv10_2_loc = nn.Conv2d(256, num_boxes["conv10_2"]*4, kernel_size= (3, 3), padding= 1)
        self.conv11_2_loc = nn.Conv2d(256, num_boxes["conv11_2"]*4, kernel_size= (3, 3), padding= 1)
        
        self.conv4_3_cls = nn.Conv2d(512, num_boxes["conv4_3"]*num_classes, kernel_size= (3, 3), padding= 1)
        self.conv7_cls = nn.Conv2d(1024, num_boxes["conv7"] * num_classes, kernel_size= (3, 3), padding= 1)
        self.conv8_2_cls = nn.Conv2d(512, num_boxes["conv8_2"]*num_classes, kernel_size= (3, 3), padding= 1)
        self.conv9_2_cls = nn.Conv2d(256, num_boxes["conv9_2"]*num_classes, kernel_size= (3, 3), padding= 1)
        self.conv10_2_cls = nn.Conv2d(256, num_boxes["conv10_2"]*num_classes, kernel_size= (3, 3), padding= 1)
        self.conv11_2_cls = nn.Conv2d(256, num_boxes["conv11_2"]*num_classes, kernel_size= (3, 3), padding= 1)
        
        self.weights_init()
        
    def forward(self, conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out):
        batch_size = conv4_3_out.size(0)
        
        l_conv4_3 = self.conv_4_3_loc(conv4_3_out)                #(N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()    #(N, 38, 38, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)             #(N, 5776, 4)
        
        l_conv7 = self.conv7_loc(conv7_out)                   #(N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()    #(N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)             #(N, 2166, 4)
        
        l_conv8_2 = self.conv8_2_loc(conv8_2_out)                 #(N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()    #(N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)             #(N, 600, 4)
        
        l_conv9_2= self.conv9_2_loc(conv9_2_out)                  #(N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()    #(N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)             #(N, 150, 4)
        
        l_conv10_2 = self.conv10_2_loc(conv10_2_out)                #(N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()    #(N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)             #(N, 36, 4)
        
        l_conv11_2 = self.conv11_2_loc(conv11_2_out)                #(N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()    #(N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)             #(N, 4, 4)

        c_conv4_3 = self.conv4_3_cls(conv4_3_out)                      #(N, 4*classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()         #(N, 38, 38, 4*classes)
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.num_classes)   #(N, 5776, classes )
        
        c_conv7 = self.conv7_cls(conv7_out)                        #(N, 6*classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()         #(N, 19, 19, 6*classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.num_classes)   #(N, 2166, classes)     
        
        c_conv8_2 = self.conv8_2_cls(conv8_2_out)                       #(N, 6*clases, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()          #(N, 10, 10, 6*classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.num_classes)    #(N, 600, classes)
        
        c_conv9_2 = self.conv9_2_cls(conv9_2_out)                       #(N, 6*classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()          #(N, 5, 5, 6*classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.num_classes)    #(N, 150, classes)
        
        c_conv10_2 = self.conv10_2_cls(conv10_2_out)                    #(N, 4*classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()        #(N, 3, 3, 4*classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.num_classes)  #(N, 36, classes)
        
        c_conv11_2 = self.conv11_2_cls(conv11_2_out)                      #(N, 4*classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()          #(N, 1, 1, 4*classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.num_classes)    #(N, 4, classes)

        
        #(N, 8732, 4)
        locs_pred = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim= 1)
 
        #(N, 8732, classes)
        cls_pred = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim= 1)    

        return locs_pred, cls_pred
    
    def weights_init(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)