import sys
import os
from os import path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../..")))

import torch
from torch import Tensor, sigmoid
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Callable, Optional, Tuple, Union
from multimodn.multimodn import get_performance_metrics

class HAIMDecoder(nn.Module):    
    def __init__(
            self,            
            n_features: int,
            hidden_layers: Tuple[int],
            n_classes: int = 2,
            hidden_activation: Callable = F.relu,
            output_activation: Callable = sigmoid,
            device: Optional[torch.device] = None,):
        super().__init__()            
            
        self.hidden_activation = hidden_activation       
        self.output_activation = output_activation
        
        dim_layers = [n_features ] + list(hidden_layers) + [n_classes, ]
        
        self.layers = nn.ModuleList()
        
        for i, (in_dim, out_dim) in enumerate(zip(dim_layers, dim_layers[1:])): 
            self.layers.append(nn.Linear(in_dim, out_dim, device=device))                     
                    
    def forward(self, x: Tensor) -> Tensor:         
        for layer in self.layers[0:-1]:
            x = self.hidden_activation(layer(x))                
        x = self.output_activation(self.layers[-1](x))        
        return x

class HAIM(nn.Module):
    def __init__(
            self,
            decoder: HAIMDecoder,          
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = decoder
        self.to(self.device)  
        
    def train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: Optimizer,
            criterion: Union[nn.Module, Callable],           
            last_epoch: bool = False,          
            
    ) -> None:

        self.train()

        n_batches = len(train_loader)        

        err_loss_epoch = 0
 
        n_correct_epoch = 0
    
        n_samples_epoch = 0

        for batch_idx, batch in enumerate(train_loader):
            data, target = list(batch)[:2]     


            batch_size = target.shape[0]
            n_samples_epoch += batch_size

            err_loss = 0
            
            target = target.type(torch.LongTensor)
            target = target.to(self.device)
            data = data.to(self.device)
            target = target[:,0]
 
            optimizer.zero_grad()        

            output_decoder = self.decoder(data)
            _, prediction = torch.max(output_decoder, dim=1)                

            err_loss = criterion(output_decoder, target)
            n_correct_epoch += sum(prediction == target).float()

            err_loss.backward()
            optimizer.step()

            err_loss_epoch += err_loss.cpu().detach().numpy()       


        err_loss_epoch /= n_batches       
        accuracy_epoch = n_correct_epoch / n_samples_epoch
 
        if last_epoch:  
            return self.test(train_loader, criterion)   
        
    def test(
            self,
            test_loader: DataLoader,
            criterion: Union[nn.Module, Callable],
    ):
        self.eval()

        n_batches = len(test_loader)
        n_samples_prediction = 0

        err_loss_prediction = 0
        n_correct_prediction = 0  

        with torch.no_grad():

            for batch_ind, batch in enumerate(test_loader):
                data, target = (list(batch))[:2]
                
                batch_size = target.shape[0]
                
                n_samples_prediction += batch_size                
                err_loss = 0               
                target = target.type(torch.LongTensor)
                target = target.to(self.device)
                data = data.to(self.device)
                target = target[:,0]
                
                if batch_ind == 0:  
                    target_decoder_epoch = target.cpu().detach()
                else:
                    target_decoder_epoch = torch.cat((target_decoder_epoch, target.cpu().detach()), dim = 0)                       
                      
                output_decoder = self.decoder(data)
                _, prediction = torch.max(output_decoder, dim=1)
                err_loss = criterion(output_decoder, target)
                n_correct_prediction += sum(
                    prediction == target).float()                               

                err_loss_prediction += err_loss.cpu().detach().numpy()
                if batch_ind == 0:
                    output_decoder_epoch = output_decoder.cpu().detach()
                else:
                    output_decoder_epoch= torch.cat((output_decoder_epoch, output_decoder.cpu().detach()), dim = 0)           
        
        err_loss_prediction /= n_batches
        accuracy_prediction = n_correct_prediction / n_samples_prediction
    
        _ , prediction_epoch = torch.max(output_decoder_epoch, dim=1)        
       
        return get_performance_metrics(target_decoder_epoch, prediction_epoch, output_decoder_epoch[:,1])
    
    
    def predict(
            self,
            test_loader: DataLoader,           
    ):
        self.eval()
 
        with torch.no_grad():

            for batch_ind, batch in enumerate(test_loader):
                data, target = (list(batch))[:2]
                
                batch_size = target.shape[0]              
                          
                target = target.type(torch.LongTensor)
                target = target.to(self.device)
                data = data.to(self.device)
                target = target[:,0]
                
                if batch_ind == 0:  
                    target_decoder_epoch = target.cpu().detach()
                else:
                    target_decoder_epoch = torch.cat((target_decoder_epoch, target.cpu().detach()), dim = 0)                      
                      
                output_decoder = self.decoder(data)
                _, prediction = torch.max(output_decoder, dim=1)                        

                if batch_ind == 0:
                    output_decoder_epoch = output_decoder.cpu().detach()
                else:
                    output_decoder_epoch= torch.cat((output_decoder_epoch, output_decoder.cpu().detach()), dim = 0)           
     
       
        return output_decoder_epoch, target_decoder_epoch
