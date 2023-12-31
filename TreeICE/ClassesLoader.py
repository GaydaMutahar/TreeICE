'''
@Author: Zhang Ruihan
@Date: 2019-10-28 01:01:52
@LastEditors: Gayda Mutahar
@LastEditTime: 2021-05-05
@Description: Add the datasetloader functions for CUB_200_2011 dataset
'''
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
import torch
import keras

import zipfile
from torchvision import transforms as T
import requests
from torch.utils.data import DataLoader

import torchvision.datasets as dset

class DataLoader():
    def set_target():
        return 

    def load_classes():
        return 

    def load_val():
        return

    def load_all():
        return

class ImageNetClassesLoader(DataLoader):

    def _load_class_name(self):
        labels = np.loadtxt(self.path + '/'+ self.label_path, str, delimiter='\t')
        classes = []
        for c in labels:
            temp = c.partition(" ")
            p = self.path + "/train/" + temp[0]
            if os.path.exists(p):
                classes.append([temp[0],temp[-1]])
        return classes
    
    # dataset ILSVRC2012, with "synset_words.txt" contains all classes name to load training set.
    def __init__(self,preprocess_input = None,targetNos = None,path = './dataset/t-imagenet',label_path = "words.txt",useqload = True,target_size = [224,224],package="torch"):
        self.path = path
        self.label_path = label_path
        self.target_size = target_size
        self.preprocess_input = preprocess_input
        self.package = package

        self.classes = self._load_class_name()
        print(self.classes)
      
        self.Name2ID = {n:k for k,n in self.classes}
        self.ID2Name = {k:n for k,n in self.classes}
        print(self.ID2Name)
    
        self.No2ID = [k for k,n in self.classes]
        print(self.No2ID)
        
        self.ID2No = {v[0]:i for i,v in enumerate(self.classes)}
        print(self.ID2No)
     
        self.Name2No = {v[1]:i for i,v in enumerate(self.classes)}
        self.No2Name = [n for k,n in self.classes]

        self.targetNos = targetNos

        self.useqload = useqload

        if not os.path.exists(self.path+"/qtrain"):
            os.mkdir(self.path+"/qtrain")
        
        self.val = self._load_val_idx()

    def set_target(self,targetNos = None):
        self.targetNos = targetNos
        
    def qsave(self,Nos,fname = "qtrain"):
        print ("qsaving...",Nos)
        for i in tqdm(Nos):
            if not os.path.exists(self.path+"/{}/{}.npy".format(fname,self.No2ID[i])):
                temp_pre = self.preprocess_input
                self.preprocess_input = None
                np.save(self.path+"/{}/{}.npy".format(fname,self.No2ID[i]),self._load_single_class(i))
                self.preprocess_input = temp_pre

    def _qload(self,No,fname = "qtrain"):
        if os.path.exists(self.path+"/{}/{}.npy".format(fname,self.No2ID[No])):
            X,sizes = np.load(self.path+"/{}/{}.npy".format(fname,self.No2ID[No]))
            if self.preprocess_input is not None:
                for i in range(X.shape[0]):
                    if self.package == "torch":
                        tx = X[i,...]
                        tx = np.transpose(tx,(2,0,1)) #to channel_first
                        tx = self.preprocess_input(torch.from_numpy(tx)).numpy()
                        tx = np.transpose(tx,(1,2,0)) #to channel_last
                        X[i,...] = tx
                    else:
                        X[i,...] = self.preprocess_input(X[i,...])
            return X,sizes
        return None
    
    def _load_val_idx(self):
        labels = np.loadtxt(self.path + '/val/val_annotations.txt', str, delimiter='\t')
        
        idx = []
        for c in labels:
            temp = c.partition(" ")
            print(temp)
            idx.append(temp[2])
        return np.array(idx)
   
    
    def _load_single_class(self,No,p=1):

        if self.useqload:
            qres = self._qload(No)
            if qres is not None:
                X,sizes = qres
                l = X.shape[0]
                if p<=1:
                    l = int(p*l)
                if p>1:
                    l = min(l,p)
                return X[:l],sizes[:l]

        ID = self.No2ID[No]
       
        path = self.path+'/train/'+ID#+'/images'
        
        imageNames = os.listdir(path)
        
        l = len(imageNames)
        if p <= 1:
            p = int(p*l)
        if p > 1:
            p = min(l,p)
        dataset = np.zeros([p]+self.target_size+[3])
        img_sizes = np.zeros([p,2])

        for i,imageName in enumerate(imageNames):
            if i == p:
                break
            img_path = path+'/'+imageName
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(transforms.Resize(self.target_size)(img))
            if self.preprocess_input is not None:
                if self.package == "torch":
                    x = self.preprocess_input(x).numpy()
                    x = np.transpose(x,(1,2,0)) #to channel_last
                else:
                    x = np.transpose(x.numpy(),(1,2,0))
                    x = self.preprocess_input(x)
            else:
                x = x.numpy()
                x = np.transpose(x,(1,2,0)) #b
            img_sizes[i,:] = img.size
            dataset[i,...] = x
        return dataset, img_sizes
    
        
    def load_train(self,Nos,p=1,require_sizes = False):
        print("******", Nos)
        if not type(Nos) == list:
            Nos = [Nos]
        print ("totally {} classes".format(len(Nos)))
        for i,No in enumerate(Nos):
            x, sizes = self._load_single_class(No,p)
            y = np.ones((x.shape[0],))*No
            if require_sizes:
                yield (No,x,y.astype(int),sizes)
            else:
                yield (No,x,y.astype(int))
                
          #  print(x,y)
        
    def load_val(self, Nos,require_sizes = False):
        if not type(Nos) == list:
            Nos = [Nos]
            
        path = self.path+'/val/images'
        dataset_x = []
        dataset_y = []
        all_sizes = []
        for n in Nos:
            print("self.val", self.val)
            print("n", n)
            n=n
            idx = np.where(self.val==n)[0]
            print("idx", idx)
            l = len(idx)
            X = np.zeros([l]+self.target_size+[3])
            img_sizes = np.zeros([l,2])
            for i,no in enumerate(list(idx)):
                img_path = path+'/'+'val_{:08d}.JPEG'.format(no+1)
                img = Image.open(img_path).convert("RGB")
                x = transforms.ToTensor()(transforms.Resize(self.target_size)(img))
                if self.preprocess_input is not None:
                    if self.package == "torch":
                        x = self.preprocess_input(x).numpy()
                        x = np.transpose(x,(1,2,0))
                    else:
                        x = np.transpose(x.numpy(),(1,2,0))
                        x = self.preprocess_input(x)
                else:
                    x = x.numpy()
                    x = np.transpose(x,(1,2,0)) #b
                img_sizes[i,:] = img.size
                X[i,...] = x
            dataset_x.append(X)
            y = np.ones((X.shape[0],))*n
            dataset_y.append(y)
            all_sizes.append(img_sizes)
            #yield (x,y)
        if require_sizes:
            return dataset_x,dataset_y,all_sizes
        else:
            return dataset_x,dataset_y


    def load_all(self,concatenate = True,require_sizes = False):
        if self.targetNos is None:
            return None
        
        X = []
        y = []
        X_sizes = []
        for No, tx, ty, tsize in self.load_train(self.targetNos,require_sizes = True):
            X.append(tx)
            y.append(ty)
            X_sizes.append(tsize)
        if concatenate:
            X = np.concatenate(X)
            y = np.concatenate(y)
            X_sizes = np.concatenate(X_sizes)
        else:
            X = np.array(X)
            y = np.array(y)
            X_sizes = np.array(X_sizes)

        tX,ty,tX_sizes = self.load_val(self.targetNos,require_sizes = True)
        if concatenate:
            tX = np.concatenate(tX)
            ty = np.concatenate(ty)
            tX_sizes = np.concatenate(tX_sizes)
        else:
            tX = np.array(tX)
            ty = np.array(ty)
            tX_sizes = np.array(tX_sizes)

        if require_sizes:
            return (X,y,X_sizes),(tX,ty,tX_sizes)
        else:
            return (X,y),(tX,ty)
    

class CUBBirdClassesLoader(DataLoader):

    def _load_class_name(self):
        labels = np.loadtxt(self.path + '/'+ self.label_path, str, delimiter='\t')
        classes = []
        for c in labels:
            temp = c.partition(" ")
            classes.append([temp[-1],temp[-1]])
        return classes
    
    def __init__(self,preprocess_input = None,targetNos = None,path = './dataset/CUB_200_2011/data',label_path = "classes.txt",useqload = True,target_size = [224,224],package="torch"):
        self.path = path
        self.label_path = label_path
        self.target_size = target_size
        self.package = package
        self.preprocess_input = preprocess_input
        

        self.classes = self._load_class_name()
        self.Name2ID = {n:k for k,n in self.classes}
        self.ID2Name = {k:n for k,n in self.classes}
        self.No2ID = [k for k,n in self.classes]
        self.ID2No = {v[0]:i for i,v in enumerate(self.classes)}
        self.Name2No = {v[1]:i for i,v in enumerate(self.classes)}
        
        self.No2Name = [n for k,n in self.classes]

        self.targetNos = targetNos

        self.useqload = useqload

        if not os.path.exists(self.path+"/qtrain"):
            os.mkdir(self.path+"/qtrain")
        
    def set_target(self,targetNos):
        self.targetNos = targetNos
        
    def qsave(self,Nos,fname = "qtrain"):
        print ("qsaving...",Nos)
        for i in tqdm(Nos):
            if not os.path.exists(self.path+"/{}/{}.npy".format(fname,self.No2ID[i])):
                temp_pre = self.preprocess_input
                self.preprocess_input = None
                np.save(self.path+"/{}/{}.npy".format(fname,self.No2ID[i]),self._load_single_class(i))
                self.preprocess_input = temp_pre

    def _qload(self,No,fname = "qtrain"):
        if os.path.exists(self.path+"/{}/{}.npy".format(fname,self.No2ID[No])):
            X,sizes = np.load(self.path+"/{}/{}.npy".format(fname,self.No2ID[No]))
            if self.preprocess_input is not None:
                for i in range(X.shape[0]):
                    if self.package == "torch":
                        tx = X[i,...]
                        tx = np.transpose(tx,(2,0,1)) #to channel_first
                        tx = self.preprocess_input(torch.from_numpy(tx)).numpy()
                        tx = np.transpose(tx,(1,2,0)) #to channel_last
                        X[i,...] = tx
                    else:
                        X[i,...] = self.preprocess_input(X[i,...])
            return X,sizes
        return None
        
    def _load_single_class(self,No,tpath = 'data/train',p=1):

        if self.useqload:
            qres = self._qload(No)
            if qres is not None:
                X,sizes = qres
                l = X.shape[0]
                if p<=1:
                    l = int(p*l)
                if p>1:
                    l = min(l,p)
                return X[:l],sizes[:l]

        ID = self.No2ID[No]
        print (self.path,tpath,ID)
        path = self.path+'/'+tpath+'/'+str(ID)
        imageNames = os.listdir(path)
        l = len(imageNames)
        if p <= 1:
            p = int(p*l)
        if p > 1:
            p = min(l,p)
        dataset = np.zeros([p]+self.target_size+[3])
        img_sizes = np.zeros([p,2])

        for i,imageName in enumerate(imageNames):
            if i == p:
                break
            img_path = path+'/'+imageName
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(transforms.Resize(self.target_size)(img))
            if self.preprocess_input is not None:
                if self.package == "torch":
                    x = self.preprocess_input(x).numpy()
                    x = np.transpose(x,(1,2,0)) #to channel_last
                else:
                    x = np.transpose(x.numpy(),(1,2,0))
                    x = self.preprocess_input(x)
            else:
                x = x.numpy()
                x = np.transpose(x,(1,2,0)) #b
            img_sizes[i,:] = img.size
            dataset[i,...] = x
            
        return dataset, img_sizes
    
        
    def load_train(self,Nos,p=1,require_sizes = False):
        if not type(Nos) == list:
            Nos = [Nos]
        print ("totally {} classes".format(len(Nos)))
        for i,No in enumerate(Nos):
            x, sizes = self._load_single_class(No,p=p)
            y = np.ones((x.shape[0],))*No
            if require_sizes:
                yield (No,x,y.astype(int),sizes)
                
            else:
                yield (No,x,y.astype(int))
            
        #return dataset_x,dataset_y
        
    def load_val(self, Nos,require_sizes = False):
        if not type(Nos) == list:
            Nos = [Nos]
        print ("totally {} classes".format(len(Nos)))

        for i,No in enumerate(Nos):
            x, sizes = self._load_single_class(No,p=1,tpath = 'data/test')
            y = np.ones((x.shape[0],))*No
            if require_sizes:
                yield (No,x,y.astype(int),sizes)
                
            else:
                yield (No,x,y.astype(int))
            


    def load_all(self,concatenate = True,require_sizes = False):
        if self.targetNos is None:
            return None
        
        X = []
        y = []
        X_sizes = []
        for No, tx, ty, tsize in self.load_train(self.targetNos,require_sizes = True):
            X.append(tx)
            y.append(ty)
            X_sizes.append(tsize)
        if concatenate:
            X = np.concatenate(X)
            y = np.concatenate(y)
            X_sizes = np.concatenate(X_sizes)
        else:
            X = np.array(X)
            y = np.array(y)
            X_sizes = np.array(X_sizes)
        
        
        tX = []
        ty = []
        tX_sizes = []
        for n, xv,yv,v_sizes in self.load_val(self.targetNos,require_sizes = True):
            tX.append(xv)
            ty.append(yv)
            tX_sizes.append(v_sizes)
                 
        if concatenate:
            tX = np.concatenate(tX)
            ty = np.concatenate(ty)
            tX_sizes = np.concatenate(tX_sizes)
        else:
            tX = np.array(tX)
            ty = np.array(ty)
            tX_sizes = np.array(tX_sizes)

        if require_sizes:
            return (No, X,y,X_sizes),(n,tX,ty,tX_sizes)
        else:
            return (No,X,y),(n,tX,ty)
    



class MNISTLoader(DataLoader):

    
    def __init__(self,channel_first = False,root = "./dataset"):
        self.classNum = 10
        self.Channel = 1
        self.img_rows = 28
        self.img_cols = 28
        
        classNum = self.classNum 
        Channel = self.Channel 
        img_rows = self.img_rows
        img_cols = self.img_cols 
        
        #(X,y),(tX,ty) = keras.datasets.mnist.load_data()
        data_train = dset.MNIST(root = root, train = True,  download = True)
        X,y = data_train.data.numpy() , data_train.targets.numpy()

        data_test = dset.MNIST(root=root, train = False)
        tX,ty = data_test.data.numpy() , data_test.targets.numpy()

        
        X = X.reshape(X.shape[0], img_rows, img_cols, Channel)
        tX = tX.reshape(tX.shape[0], img_rows, img_cols, Channel)
        input_shape = (img_rows, img_cols, Channel)

        X = X.astype("float32")
        X = X-X.min()
        X = X / X.max()
        tX = tX.astype("float32")
        tX = tX - tX.min()
        tX = tX / tX.max()
        
        #y = keras.utils.to_categorical(y,classNum)
        #ty = keras.utils.to_categorical(ty,classNum)

        y = np.eye(classNum, dtype='uint8')[y]
        ty = np.eye(classNum, dtype='uint8')[ty]

        if channel_first:
            X = np.transpose(X,(0,3,1,2))
            tX = np.transpose(tX,(0,3,1,2))
        
        self.X = X
        self.y = y
        self.tX = tX
        self.ty = ty
        
        ydict = []
        tydict = []
        for i in range(10):
            ydict.append(np.array([j for j in range(y.shape[0]) if y[j][i]==1]))
            tydict.append(np.array([j for j in range(ty.shape[0]) if ty[j][i]==1]))
        
        self.ydict = ydict
        self.tydict = tydict
        
        self.Name2ID = list(range(10))
        self.ID2Name = list(range(10))
        self.No2ID = list(range(10))
        self.ID2No = list(range(10))
        self.Name2No = list(range(10))
        self.No2Name = list(range(10))
        
        
    def load_train(self,Nos=[0]):

        dataset_x = []
        dataset_y = []
        for i in Nos:
            x = self.X[self.ydict[i]]
            y = np.ones((x.shape[0],))*i
            yield i,x,y
            #dataset_x.append(x)
            #dataset_y.append(y)
        #return dataset_x,dataset_y
        
    def load_val(self, Nos=[0]):
        dataset_x = []
        dataset_y = []
        for i in Nos:
            x = self.tX[self.tydict[i]]
            y = np.ones((x.shape[0],))*i
            dataset_x.append(x)
            dataset_y.append(y)
        return dataset_x,dataset_y
        return dataset_x,dataset_y
        
    def load_all(self):
        return (self.X,self.y),(self.tX,self.ty)
    
class MRPClassesLoader(DataLoader):
    
    def _load_class_name(self):
        labels = np.loadtxt(self.path + '/'+ self.label_path, str, delimiter='\t')
        classes = []
        for c in labels:
            temp = c.partition(" ")
            classes.append([temp[-1],temp[-1]])
        return classes
    
    def __init__(self,preprocess_input = None,targetNos = None,path = './MRP/MRI_32_Images/data_00',label_path = "classes.txt",useqload = True,target_size = [150,150],package="torch"):
        self.path = path
        self.label_path = label_path
        self.target_size = target_size
        self.package = package
        self.preprocess_input = preprocess_input
        

        self.classes = self._load_class_name()
        self.Name2ID = {n:k for k,n in self.classes}
        self.ID2Name = {k:n for k,n in self.classes}
        self.No2ID = [k for k,n in self.classes]
        self.ID2No = {v[0]:i for i,v in enumerate(self.classes)}
        self.Name2No = {v[1]:i for i,v in enumerate(self.classes)}
        
        self.No2Name = [n for k,n in self.classes]

        self.targetNos = targetNos

        self.useqload = useqload

        if not os.path.exists(self.path+"/qtrain"):
            os.mkdir(self.path+"/qtrain")
        
    def set_target(self,targetNos):
        self.targetNos = targetNos
        
    def qsave(self,Nos,fname = "qtrain"):
        print ("qsaving...",Nos)
        for i in tqdm(Nos):
            if not os.path.exists(self.path+"/{}/{}.npy".format(fname,self.No2ID[i])):
                temp_pre = self.preprocess_input
                self.preprocess_input = None
                np.save(self.path+"/{}/{}.npy".format(fname,self.No2ID[i]),self._load_single_class(i))
                self.preprocess_input = temp_pre

    def _qload(self,No,fname = "qtrain"):
        if os.path.exists(self.path+"/{}/{}.npy".format(fname,self.No2ID[No])):
            X,sizes = np.load(self.path+"/{}/{}.npy".format(fname,self.No2ID[No]))
            if self.preprocess_input is not None:
                for i in range(X.shape[0]):
                    if self.package == "torch":
                        tx = X[i,...]
                        tx = np.transpose(tx,(2,0,1)) #to channel_first
                        tx = self.preprocess_input(torch.from_numpy(tx)).numpy()
                        tx = np.transpose(tx,(1,2,0)) #to channel_last
                        X[i,...] = tx
                    else:
                        X[i,...] = self.preprocess_input(X[i,...])
            return X,sizes
        return None
        
    def _load_single_class(self,No,tpath = 'train_',p=1):

        if self.useqload:
            qres = self._qload(No)
            if qres is not None:
                X,sizes = qres
                l = X.shape[0]
                if p<=1:
                    l = int(p*l)
                if p>1:
                    l = min(l,p)
                return X[:l],sizes[:l]

        ID = self.No2ID[No]
        print (self.path,tpath,ID)
        path = self.path+'/'+tpath+'/'+str(ID)
        imageNames = os.listdir(path)
        l = len(imageNames)
        if p <= 1:
            p = int(p*l)
        if p > 1:
            p = min(l,p)
        dataset = np.zeros([p]+self.target_size+[3])
        img_sizes = np.zeros([p,2])

        for i,imageName in enumerate(imageNames):
            if i == p:
                break
            img_path = path+'/'+imageName
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(transforms.Resize(self.target_size)(img))
            if self.preprocess_input is not None:
                if self.package == "torch":
                    x = self.preprocess_input(x).numpy()
                    x = np.transpose(x,(1,2,0)) #to channel_last
                else:
                    x = np.transpose(x.numpy(),(1,2,0))
                    x = self.preprocess_input(x)
            else:
                x = x.numpy()
                x = np.transpose(x,(1,2,0)) #b
            img_sizes[i,:] = img.size
            dataset[i,...] = x
        print("img_sizes", img_sizes)   
        return dataset, img_sizes
    
        
    def load_train(self,Nos,p=1,require_sizes = False):
        if not type(Nos) == list:
            Nos = [Nos]
        print ("totally {} classes".format(len(Nos)))
        for i,No in enumerate(Nos):
            x, sizes = self._load_single_class(No,p=p)
            y = np.ones((x.shape[0],))*No
            if require_sizes:
                yield (No,x,y.astype(int),sizes)
                
            else:
                yield (No,x,y.astype(int))
            
        #return dataset_x,dataset_y
        
    def load_val(self, Nos,require_sizes = False):
        if not type(Nos) == list:
            Nos = [Nos]
        print ("totally {} classes".format(len(Nos)))

        for i,No in enumerate(Nos):
            x, sizes = self._load_single_class(No,p=1,tpath = 'val')
            y = np.ones((x.shape[0],))*No
            if require_sizes:
                yield (No,x,y.astype(int),sizes)
                
            else:
                yield (No,x,y.astype(int))
            


    def load_all(self,concatenate = True,require_sizes = False):
        if self.targetNos is None:
            return None
        
        X = []
        y = []
        X_sizes = []
        for No, tx, ty, tsize in self.load_train(self.targetNos,require_sizes = True):
            X.append(tx)
            y.append(ty)
            X_sizes.append(tsize)
        if concatenate:
            X = np.concatenate(X)
            y = np.concatenate(y)
            X_sizes = np.concatenate(X_sizes)
        else:
            X = np.array(X)
            y = np.array(y)
            X_sizes = np.array(X_sizes)
        
        
        tX = []
        ty = []
        tX_sizes = []
        for n, xv,yv,v_sizes in self.load_val(self.targetNos,require_sizes = True):
            tX.append(xv)
            ty.append(yv)
            tX_sizes.append(v_sizes)
                 
        if concatenate:
            tX = np.concatenate(tX)
            ty = np.concatenate(ty)
            tX_sizes = np.concatenate(tX_sizes)
        else:
            tX = np.array(tX)
            ty = np.array(ty)
            tX_sizes = np.array(tX_sizes)

        if require_sizes:
            return (No, X,y,X_sizes),(n,tX,ty,tX_sizes)
        else:
            return (No,X,y),(n,tX,ty)
    
    
        
    
class CIFAR10ClassesLoader(DataLoader):

    def __init__(self,channel_first = False,root = "./dataset"):
        self.classNum = 10
        self.Channel = 1
        self.img_rows = 28
        self.img_cols = 28
        
        classNum = self.classNum 
        Channel = self.Channel 
        img_rows = self.img_rows
        img_cols = self.img_cols 
        
        #(X,y),(tX,ty) = keras.datasets.mnist.load_data()
        data_train = dset.CIFAR10(root = root, train = True,  download = True)
        X,y = data_train.data , data_train.targets

        data_test = dset.CIFAR10(root=root, train = False)
        tX,ty = data_test.data , data_test.targets

        
     #   X = X.reshape(X.shape[0], img_rows, img_cols, Channel)
      #  tX = tX.reshape(tX.shape[0], img_rows, img_cols, Channel)
        input_shape = (img_rows, img_cols, Channel)

        X = X.astype("float32")
        X = X-X.min()
        X = X / X.max()
        tX = tX.astype("float32")
        tX = tX - tX.min()
        tX = tX / tX.max()
        
        #y = keras.utils.to_categorical(y,classNum)
        #ty = keras.utils.to_categorical(ty,classNum)

        y = np.eye(classNum, dtype='uint8')[y]
        ty = np.eye(classNum, dtype='uint8')[ty]

        if channel_first:
            X = np.transpose(X,(0,3,1,2))
            tX = np.transpose(tX,(0,3,1,2))
        
        self.X = X
        self.y = y
        self.tX = tX
        self.ty = ty
        
        ydict = []
        tydict = []
        for i in range(10):
            ydict.append(np.array([j for j in range(y.shape[0]) if y[j][i]==1]))
            tydict.append(np.array([j for j in range(ty.shape[0]) if ty[j][i]==1]))
        
        self.ydict = ydict
        self.tydict = tydict
        
        self.Name2ID = list(range(10))
        print("self.Name2ID", self.Name2ID)
        self.ID2Name = list(range(10))
        self.No2ID = list(range(10))
        self.ID2No = list(range(10))
        self.Name2No = list(range(10))
        print("Name2No", self.Name2ID)
        self.No2Name = list(range(10))
        
        
    def load_train(self,Nos=[0]):

        dataset_x = []
        dataset_y = []
        for i in Nos:
            x = self.X[self.ydict[i]]
            y = np.ones((x.shape[0],))*i
            yield i,x,y
            #dataset_x.append(x)
            #dataset_y.append(y)
        #return dataset_x,dataset_y
        
    def load_val(self, Nos=[0]):
        dataset_x = []
        dataset_y = []
        for i in Nos:
            x = self.tX[self.tydict[i]]
            y = np.ones((x.shape[0],))*i
            dataset_x.append(x)
            dataset_y.append(y)
        return dataset_x,dataset_y
        return dataset_x,dataset_y
        
    def load_all(self):
        return (self.X,self.y),(self.tX,self.ty)
        