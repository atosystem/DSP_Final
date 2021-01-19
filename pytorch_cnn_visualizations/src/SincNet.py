import torch.nn as nn
class SincNet_Model(nn.Module):
    
    def __init__(self, CNN, DNN1, DNN2):
        """
        docstring
        """
        super(SincNet_Model,self).__init__()
        self.CNN = CNN
        self.DNN1 = DNN1
        self.DNN2 = DNN2
        # self.CNN.eval()
        # self.DNN1.eval()
        # self.DNN2.eval()
    
    

    def forward(self, x):
    
        x1 = self.CNN(x)
        x2 = self.DNN1(x1)
        x3 = self.DNN2(x2)
        
        return x3