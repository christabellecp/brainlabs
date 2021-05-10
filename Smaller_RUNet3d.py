from __future__ import print_function, division

from CogDataset3d import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# pytorch stuff
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(3)


class Smaller_RUNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Smaller_RUNet3D, self).__init__()

        self.cb1 = contraction_block3d(in_ch=1, out_ch=16)
        self.cb2 = contraction_block3d(in_ch=16, out_ch=32)
        self.cb3 = contraction_block3d(in_ch=32, out_ch=64)
        self.convb = conv_block3d(in_ch=64, out_ch=128)
        self.regb = regression_block3d(in_ch=128, out_ch=64)  
        #self.dropout = nn.Dropout(0.10)
        self.linear = nn.Linear(64*8*8*8, 1) 
        #self.regb2 = regression_block3d(in_ch=128, out_ch=64)  
        #self.dropout2 = nn.Dropout(0.30)
        #self.linear2 = nn.Linear(64*8*8*8, 1) 
        self.exb1 = expansion_block3d(in_ch=128, out_ch=64)
        self.exb2 = expansion_block3d(in_ch=64, out_ch=32)
        self.exb3 = expansion_block3d(in_ch=32, out_ch=16)
        self.final_conv = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3, padding=1, stride=1)
 
        
    def forward(self, X):
        cb1_out, out1 = self.cb1(X)
        cb2_out, out2 = self.cb2(out1)
        cb3_out, out3 = self.cb3(out2)
        out4 = self.convb(out3)
        regb_out = self.regb(out4)
        #drp_out = self.dropout(regb_out)
        reg_x =  regb_out.view(-1,64*8*8*8) 
        reg_out = self.linear(reg_x) 
      #  regb_out2 = self.regb(out4)
      #  drp_out2 = self.dropout(regb_out2)
     #   reg_x2 =  drp_out2.view(-1,64*8*8*8) 
     #   reg_out2 = self.linear(reg_x2)  
        out5 = self.exb1(cb3_out, out4)
        out6 = self.exb2(cb2_out, out5)
        out7 = self.exb3(cb1_out, out6)
        seg_out = self.final_conv(out7)
        #return reg_out,reg_out2, seg_out
        return reg_out, seg_out
    



class conv_block3d(nn.Module):
    def __init__(self, in_ch, out_ch, same=True):
        super(conv_block3d, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=1)  
        self.bn1 = nn.BatchNorm3d(out_ch, track_running_stats = True)                        
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, stride=1) 
        self.bn2 = nn.BatchNorm3d(out_ch, track_running_stats = True)                          
        
    def forward(self, X):
        X1 = F.relu(self.bn1(self.conv1(X)))
        X2 = F.relu(self.bn2(self.conv2(X1)))
        return X2

class regression_block3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(regression_block3d, self).__init__()
        self.convb = conv_block3d(in_ch, out_ch)
        self.poolreg = nn.MaxPool3d(kernel_size=2, stride=2)  
        self.bnreg = nn.BatchNorm3d(out_ch, track_running_stats = True)                        
    
    def forward(self, X):
        poolreg_out = self.poolreg(self.convb(X))
        bnreg_out = self.bnreg(poolreg_out)
        return bnreg_out 
    
class contraction_block3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(contraction_block3d, self).__init__()                                           
        self.convb = conv_block3d(in_ch, out_ch)              
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)            
        
    def forward(self, X):
        conv_out = self.convb.forward(X)
        maxpool_out = self.pool(conv_out)
        return conv_out,maxpool_out
    
class expansion_block3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(expansion_block3d, self).__init__()
        self.transpose_conv = nn.ConvTranspose3d(in_ch, out_ch, 2, 2) 
        self.convb = conv_block3d(in_ch, out_ch)
          
    def forward(self, contraction_out, X):
        transpose_conv_out = self.transpose_conv(X)
        concat_input = torch.cat((contraction_out,transpose_conv_out), dim=1)
        conv_out = self.convb(concat_input)
        return conv_out
    
    
