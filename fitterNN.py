from functions import *

import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Define fitter class as a NN
#Define fitter class as a NN
class fitter(nn.Module):
    def __init__(self):
        super(fitter, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 32),  
            nn.LeakyReLU(),
            nn.Linear(32, 64),  
            nn.LeakyReLU(),
            nn.Linear(64,32), 
            nn.LeakyReLU(),
            nn.Linear(32, 2)   
        )


    def forward(self, x):
        return self.layers(x)
    

#Set fitter
fit = fitter().to(device)

#Define NN criteria and optmizer
criterion = nn.MSELoss()
optimizer = optim.Adam(fit.parameters(), lr=0.01, weight_decay=1e-5)  


