'''
@Author: Zhang Ruihan
@Date: 2019-10-28 01:01:52
@LastEditors  : Gayda Mutahar
@LastEditTime : 2021-05-07 
@Description: To fix some parameters
'''

import numpy as np
import os
import keras
import torch
from ClassesLoader import *
from torch.utils.data import TensorDataset,DataLoader



class ModelWrapper:
    def __init__(self,model,batch_size = 128):
        self.model = model
        self.batch_size = batch_size
    def get_feature(self,x,layer_name):
        '''
        get feature map from a given layer
        '''
        pass
    def feature_predict(self,feature,layer_name = None):
        pass  
    def predict(self,x):
        pass
    def target_value(self,x):
        pass
    

class PytorchModelWrapper(ModelWrapper):   
    def __init__(self,
                 model,
                 layer_dict = {},
                 target = None,
                 channel_last = False,
                 input_size = [3,224,224],
                 batch_size=128):#target: (layer_name,unit_nums)
        self.layer_dict = layer_dict
        self.layer_dict.update(dict(model.named_children()))
        self.target = target
        self.channel_last = channel_last
        self.input_size = list(input_size)
        
        self.CUDA = torch.cuda.is_available()

        super().__init__(model,batch_size)

    def set_target(self,targetNos = None, target_layer = None):
        self.target = (target_layer,targetNos)

    def _to_tensor(self,x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        return x

    def _switch_channel_f(self,x):
        if not self.channel_last:
            if x.ndim == 3:
                x = np.transpose(x,(2,0,1))
            else:
                x = np.transpose(x,(0,3,1,2))
        return x

    def _switch_channel_b(self,x):
        if not self.channel_last:
            if x.ndim == 3:
                x = np.transpose(x,(1,2,0))
            elif x.ndim == 4:
                x = np.transpose(x,(0,2,3,1))
            elif x.ndim == 5:
                x = np.transpose(x,(0,3,4,2,1))
        return x

    def _fun(self,x,layer_in = "input",layer_out = "output"):
        #tensor cpu in cpu out

        x = x.type(torch.FloatTensor)


        in_flag = False
        if layer_in == "input":
            in_flag = True
        

        data_in = x.clone()
        if self.CUDA:
            data_in = data_in.cuda()
        data_out = []

        handles = []
        
        def hook_in(m,i,o):
            return data_in
        def hook_out(m,i,o):
            data_out.append(o)

        if layer_in == "input":
            nx = x
        else:
            handles.append(self.layer_dict[layer_in].register_forward_hook(hook_in))
            nx = torch.zeros([x.size()[0]]+self.input_size)

        if not layer_out == "output":
            handles.append(self.layer_dict[layer_out].register_forward_hook(hook_out))

        if self.CUDA:
            nx = nx.cuda()
            
        with torch.no_grad():
            nx.unsqueeze_(0)
            ny = self.model(nx)
        
        if layer_out == "output":
            data_out = ny
        else:
            data_out = data_out[0]

        data_out = data_out.cpu()

        for handle in handles:
            handle.remove() 

        return data_out

    def _batch_fn(self,x,layer_in = "input",layer_out = "output"):
        #tensor in tensor out
        out = []

        batch_size = self.batch_size
        
        if type(x) == torch.Tensor or type(x) == np.ndarray:
            x = self._to_tensor(x)

            
        l = x.shape[0]
        it_num = l // batch_size
        fr = 0
        to = 0
        for i in range(it_num):
            fr = i*batch_size
            to = (i+1)*batch_size
            nx = x[fr:to]
            out.append(self._fun(nx,layer_in,layer_out))
        nx = x[to:]
        if nx.shape[0] > 0:
            out.append(self._fun(nx,layer_in,layer_out))

        res = torch.cat(out,0)

        return res

    
    def get_feature(self,x,layer_name):
        x = self._switch_channel_f(x)
       # print("self.layer_dict", self.layer_dict)
        if layer_name not in self.layer_dict:
            return None

        nx = self._to_tensor(x)
        out = self._batch_fn(nx,layer_out = layer_name)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out

    def feature_predict(self,feature,layer_name = None):
        feature = self._switch_channel_f(feature)
        if layer_name not in self.layer_dict:
            return None

        nx = self._to_tensor(feature)
        out = self._batch_fn(nx,layer_in = layer_name)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out        


    def target_predict(self,feature,layer_name = None):
        feature = self._switch_channel_f(feature)

        if self.target is None:
            print ("No target")
            return None
        if layer_name not in self.layer_dict:
            print ("layer not found")
            return None

        target_layer,unit_nums = self.target
        nx = self._to_tensor(feature)
        
        out = self._batch_fn(nx,layer_in = layer_name,layer_out = target_layer)
        
        out = out.numpy()

        out = self._switch_channel_b(out)
        
        return out[...,unit_nums]


    def predict(self,x, targetclass = None):
        x = self._switch_channel_f(x)

        nx = self._to_tensor(x)
        out = self._batch_fn(nx)
        out = out.numpy()
        
        out = self._switch_channel_b(out)
       
        return out
    
    
class KerashModelWrapper(ModelWrapper):   
    def __init__(self,
                 model,
                 layer_dict = [],
                 target = None,
                 channel_last = False,
                 input_size = [3,224,224],
                 batch_size=128):#target: (layer_name,unit_nums)
        self.layer_dict = [model.get_layer(index = layer).name for layer in range(len(model.layers))]
        #self.layer_dict.append([model.get_layer(index = layer).name for layer in range(len(model.layers))])
        self.target = target
        self.channel_last = channel_last
        self.input_size = list(input_size)
        
        self.CUDA = torch.cuda.is_available()

        super().__init__(model,batch_size)

    def set_target(self,targetNos = None, target_layer = None):
        self.target = (target_layer,targetNos)

    def _to_tensor(self,x):
        if type(x) == np.ndarray:
            x = keras.from_numpy(x)
        return x

    def _switch_channel_f(self,x):
        if self.channel_last:
            if x.ndim == 3:
                x = np.transpose(x,(2,0,1))
            else:
                x = np.transpose(x,(0,3,1,2))
        return x

    def _switch_channel_b(self,x):
        if self.channel_last:
            if x.ndim == 3:
                x = np.transpose(x,(1,2,0))
            elif x.ndim == 4:
                x = np.transpose(x,(0,2,3,1))
        return x

    def _fun(self,x,layer_in = "input",layer_out = "output"):
        #tensor cpu in cpu out

        #x = x.type(torch.FloatTensor)


        in_flag = False
        if layer_in == "input":
            in_flag = True
        

        data_in = x.copy()
        if self.CUDA:
            data_in = data_in.cuda()
        data_out = []

        handles = []
        
        def hook_in(m,i,o):
            return data_in
        def hook_out(m,i,o):
            data_out.append(o)

        if layer_in == "input":
            nx = x
        else:
            handles.append(self.layer_dict[layer_in].register_forward_hook(hook_in))
            nx = torch.zeros([x.size()[0]]+self.input_size)

        if not layer_out == "output":
            handles.append(self.layer_dict[layer_out].register_forward_hook(hook_out))

        if self.CUDA:
            nx = nx.cuda()
            
        with torch.no_grad():
            ny = self.model(nx)

        #print(data_out)

        if layer_out == "output":
            data_out = ny
        else:
            data_out = data_out[0]

        data_out = data_out.cpu()

        for handle in handles:
            handle.remove() 

        return data_out

    def _batch_fn(self,x,layer_in = "input",layer_out = "output"):
        #tensor in tensor out
        out = []

        batch_size = self.batch_size
        l = x.shape[0]
        it_num = l // batch_size
        fr = 0
        to = 0
        for i in range(it_num):
            fr = i*batch_size
            to = (i+1)*batch_size
            nx = x[fr:to]
            out.append(self._fun(nx,layer_in,layer_out))
        nx = x[to:]
        if nx.shape[0] > 0:
            out.append(self._fun(nx,layer_in,layer_out))

        res = torch.cat(out,0)

        return res

    
    def get_feature(self,x,layer_name):
        x = self._switch_channel_f(x)
        print("self.layer_dict", self.layer_dict)
        if layer_name not in self.layer_dict:
            return None

        #nx = self._to_tensor(x)
        out = self._batch_fn(x,layer_out = layer_name)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out

    def feature_predict(self,feature,layer_name = None):
        feature = self._switch_channel_f(feature)
        if layer_name not in self.layer_dict:
            return None

        #nx = self._to_tensor(feature)
        out = self._batch_fn(feature,layer_in = layer_name)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out        


    def target_predict(self,feature,layer_name = None):
        feature = self._switch_channel_f(feature)

        if self.target is None:
            print ("No target")
            return None
        if layer_name not in self.layer_dict:
            print ("layer not found")
            return None

        target_layer,unit_nums = self.target
        nx = self._to_tensor(feature)
        out = self._batch_fn(nx,layer_in = layer_name,layer_out = target_layer)
        out = out.numpy()

        out = self._switch_channel_b(out)
        return out[...,unit_nums]


    def predict(self,x):
        x = self._switch_channel_f(x)

        nx = self._to_tensor(x)
        out = self._batch_fn(nx)
        out = out.numpy()
        
        out = self._switch_channel_b(out)
        return out