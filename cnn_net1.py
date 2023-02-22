import torch.nn as nn
import torch.nn.functional as F

#Class for Net1 CNN
class HW4Net (nn.Module):
    def __init__ ( self ):
        super (HW4Net , self ). __init__ ()
        self . conv1 = nn. Conv2d (3, 16 , 3)
        self . pool = nn. MaxPool2d (2, 2)
        self . conv2 = nn. Conv2d (16 , 32 , 3)
        self . fc1 = nn. Linear (6272 , 64) # output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
        self . fc2 = nn. Linear (64 , 5)    #The number is 5 because the classes are [airplane, bus, cat, dog, pizza]

    def forward (self , x):
        x = self . pool (F. relu ( self . conv1 (x)))
        x = self . pool (F. relu ( self . conv2 (x)))
        x = x. view (x.shape[0], -1)
        x = F. relu ( self .fc1(x))
        x = self .fc2(x)
        return x