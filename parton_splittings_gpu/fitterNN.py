import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define fitter class as a NN
#Define fitter class as a NN

    
class simple_fitter(nn.Module):
    def __init__(self):
        super(simple_fitter, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(4, 64),  
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Sinusoidal feature transformation
        
        return self.layers(x)



