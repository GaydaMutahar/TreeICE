from utils import *
from ClassesLoader import *
from ModelWrapper import *

from ChannelReducer import ChannelReducer, ClusterReducer
import os
import pickle

import graphviz
import pydotplus

import scipy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from torchvision import transforms
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV








FONT_SIZE = 30
CALC_LIMIT = 1e8
TRAIN_LIMIT = 50
TARGET_ERR = 0.2
MIN_STEP = 10
REDUCER_PATH = "reducer/resnet50"
USE_TRAINED_REDUCER = False


class Explainer():
    def __init__(self,
                 title = "",
                 layer_name = "",
                 classesNos = None,
                 utils = None,
                 nchannels = 3,
                 useMean = True,
                 reducer_type = "NMF",
                 n_components = 15,
                 best_n = False,
                 target_err = TARGET_ERR,
                 min_step = MIN_STEP,
                 featuretopk = 20,
                 featureimgtopk = 5,
                 epsilon = 1e-7):
        self.title = title
        self.layer_name = layer_name
        self.classesNos = classesNos

        if self.classesNos is not None:
            self.C2IDX = {c:i for i,c in enumerate(self.classesNos)}
            self.IDX2C = {i:c for i,c in enumerate(self.classesNos)}

        self.useMean = useMean
        self.reducer_type = reducer_type
        self.nchannels = nchannels
        self.featuretopk = featuretopk
        self.featureimgtopk = featureimgtopk #number of images for a feature
        self.n_components = n_components
        self.target_err = target_err
        self.best_n = best_n
        self.min_step = min_step
        self.epsilon = epsilon
        
        self.utils = utils

        self.reducer = None

        self.feature_base = []
        self.features = {}
        self.tree_model = None

        self.font = FONT_SIZE
        
    def load(self):
        title = self.title
        with open("Explainers"+"/"+title+"/"+title+".pickle","rb") as f:
            tdict = pickle.load(f)
            self.__dict__.update(tdict)
            
    def save(self):
        if not os.path.exists("Explainers"):
            os.mkdir("Explainers")
        title = self.title
        if not os.path.exists("Explainers"+"/"+title):
            os.mkdir("Explainers"+"/"+title)
        with open("Explainers"+"/"+title+"/"+title+".pickle","wb") as f:
            pickle.dump(self.__dict__,f)

    def train_model(self,model,classesLoader):
        if self.best_n:
            print ("search for best n.")
            self.n_components = self.min_step
            train_count = 0
            while train_count < TRAIN_LIMIT:
                print ("try n_component with {}".format(self.n_components))
                self.reducer = None
                self._train_reducer(model,classesLoader)
                if self.reducer_err.mean() < self.target_err:
                    self._estimate_weight(model,classesLoader)
                    return 
                self.n_components += self.min_step
                train_count += 1
        else:
            self._train_reducer(model,classesLoader)
            self._estimate_weight(model,classesLoader)
           

    def _train_reducer(self,model,classesLoader):
        X_feature = []

        print ("Training reducer:")
        print ("Loading data")

        if self.reducer is None:
            if self.reducer_type == "Cluster":
                self.reducer = ClusterReducer(n_clusters = self.n_components)
            elif self.reducer_type == "NMF":
                if len(self.classesNos) == 1 and USE_TRAINED_REDUCER:
                    target_path = REDUCER_PATH + "/{}/{}.pickle".format(self.layer_name,self.classesNos[0])
                    if os.path.exists(target_path):
                        with open(target_path,"rb") as f:
                            reducers = pickle.load(f)
                        if self.n_components in reducers:
                            self.reducer = reducers[self.n_components]['reducer']
                            print ("Reducer loaded")
                            
                if self.reducer is None:
                    self.reducer = ChannelReducer(n_components = self.n_components)
            else:
                self.reducer = ChannelReducer(n_components = self.n_components,reduction_alg = self.reducer_type)


        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            featureMaps = model.get_feature(X,self.layer_name)
            X_feature.append(featureMaps)
           
        #X_feature = np.concatenate(X_feature)
        #X_feature = np.array(X_feature)

        if not self.reducer._is_fit:
            X_feature_f = np.concatenate(X_feature)
            total = np.product(X_feature_f.shape)
            l = X_feature_f.shape[0]
            nX_feature = X_feature_f
            if total > CALC_LIMIT:
                p = CALC_LIMIT / total
                print ("Dataset too big, train with {:.2f} instances".format(p))
                idx = np.random.choice(l,int(l*p),replace = False)
                nX_feature = nX_feature[idx]

            print ("Loading complete, with size of {}".format(nX_feature.shape))
            
            print ("Training will take around a minute, please wait for a while...".format(nX_feature.shape))
            start_time = time.time()
           
            nX = self.reducer.fit(nX_feature)
            

            print ("reducer trained, spent {} s".format(time.time()-start_time))
        
        self.ncavs = self.reducer._reducer.components_
       # print("ncavs:", self.ncavs)
      
        
        reX = []
        for i in range(len(self.classesNos)):
            nX = self.reducer.transform(X_feature[i])
            reX.append(self.reducer.inverse_transform(nX))

        err = []
        for i in range(len(self.classesNos)):
            res_true = model.target_predict(X_feature[i],layer_name=self.layer_name)[:,i]
            res_recon = model.target_predict(reX[i],layer_name=self.layer_name)[:,i]
            err.append(abs(res_true-res_recon).mean(axis=0) / res_true.mean(axis=0))
         


        self.reducer_err = np.array(err)
        if type(self.reducer_err) is not np.ndarray:
            self.reducer_err = np.array([self.reducer_err])

        print ("fidelity: {}".format(self.reducer_err))

        return self.reducer_err

    def _estimate_weight(self,model,classesLoader):
        # a function to estimate the weights or concept importance (the estimated feature importance for classifier
        #targeting class k for given a feature map of the training dataset, the feature importance for NCAV is
        #irrelevant to the input feature map)
        X_feature = []

        print ("loading data")

        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            featureMaps = model.get_feature(X,self.layer_name)

            X_feature.append(featureMaps)
        X_feature = np.concatenate(X_feature)

        self.test_weight = []
        print ("estimating weight:")
        for i in tqdm(range(self.n_components)):
            ncav = self.ncavs[i,:] #learned ncav parameter 
       
            res1 =  model.target_predict(X_feature - self.epsilon * ncav,layer_name=self.layer_name)
            res2 =  model.target_predict(X_feature + self.epsilon * ncav,layer_name=self.layer_name)

            res_dif = res2 - res1
            dif = res_dif.mean(axis=0) / (2 * self.epsilon)
            if type(dif) is not np.ndarray:
                dif = np.array([dif])
            self.test_weight.append(dif)

        self.test_weight = np.array(self.test_weight)

    def save_features(self,threshold=0.5,background = 0.2,smooth = True):
        feature_path = "Explainers/"+self.title + "/feature_imgs"
      
        utils = self.utils
      
       
        if not os.path.exists(feature_path):
            os.mkdir(feature_path) 

        for idx in tqdm(self.features.keys()): 

            x,h = self.features[idx]
           
            
            #x = self.gen_masked_imgs(x,h,threshold=threshold,background = background,smooth = smooth)
            minmax = False
            if self.reducer_type == 'PCA':
                minmax = True
            x,h = self.utils.img_filter(x,h,threshold=threshold,background = background,smooth = smooth,minmax = minmax)
         
            
            nsize = self.utils.img_size.copy()
            nsize[1] = nsize[1]* self.featureimgtopk
            nimg = np.zeros(nsize)
            nh = np.zeros(nsize[:-1])
            for i in range(x.shape[0]):
                timg = utils.deprocessing(x[i])
                if timg.max()>1:
                    timg = timg / 255.0
                    timg = abs(timg)
                timg = np.clip(timg,0,1)
                nimg[:,i*self.utils.img_size[1]:(i+1)*self.utils.img_size[1],:] = timg
                nh[:,i*self.utils.img_size[1]:(i+1)*self.utils.img_size[1]] = h[i]
            fig = self.utils.contour_img(nimg,nh)
            fig.savefig(feature_path + "/"+str(idx)+".jpg",bbox_inches='tight',pad_inches=0)
           # plt.show(fig)
            #plt.close(fig)
            #plt.imsave(feature_path + "/"+str(idx)+".jpg",nimg)

    def feature_filter(self,featureMaps):
        if self.useMean:
            return featureMaps.mean(axis = (1,2))
        else:
            return featureMaps.max(axis=(1,2))
   
    
    def generate_features(self,model,classesLoader, featureIdx = None):
        featuretopk = min(self.featuretopk, self.n_components)

        imgTopk = self.featureimgtopk
        if featureIdx is None:
            featureIdx = []
            tidx = []
            w = self.test_weight
            for i,No in enumerate(self.classesNos):
                tw = w[:,i]
                tidx += tw.argsort()[::-1][:featuretopk].tolist()  #indeces of the estimated weight of each extracted concept from the target classes           
            featureIdx += list(set(tidx))
           
                  

        nowIdx = set(self.features.keys())
        featureIdx = list(set(featureIdx) - nowIdx)
        featureIdx.sort()

        if len(featureIdx) == 0:
            print ("All feature gathered")
            #return

        print ("generating features:")
        print (featureIdx)

        features = {}
        for No in featureIdx:
            features[No] = [None,None]
        
        print ("loading training data")
        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            
            featureMaps = self.reducer.transform(model.get_feature(X,self.layer_name))
           # print("featureMaps", featureMaps)
         
            X_feature = self.feature_filter(featureMaps)
           

            for No in featureIdx:
                samples,heatmap = features[No]
                idx = X_feature[:,No].argsort()[-imgTopk:]
                
                nheatmap = featureMaps[idx,:,:,No]
                nsamples = X[idx,...]
                
                if type(samples) == type(None):
                    samples = nsamples
                    heatmap = nheatmap
                else:
                    samples = np.concatenate([samples,nsamples])
                    heatmap = np.concatenate([heatmap,nheatmap])

                    nidx = self.feature_filter(heatmap).argsort()[-imgTopk:]
                    samples = samples[nidx,...]
                    heatmap = heatmap[nidx,...]
                
                features[No] = [samples,heatmap]
                
        
        for no,(x,h) in features.items():
            idx = h.mean(axis = (1,2)).argmax()
            for i in range(h.shape[0]):
                if h[i].max() == 0:
                    x[i] = x[idx]
                    h[i] = h[idx]
        
        self.features.update(features)
        self.save()
        #print("features", features)
       
        return (self.features)

####### Generate the baseline (ICE explanations) 
    def generate_image_LR_file(self,classesLoader):        
        title = self.title
        fpath = os.getcwd() + "/Explainers/"+ self.title + "/feature_imgs/"
        featopk = min(self.featuretopk,self.n_components)
        imgtopk = self.featureimgtopk
        classes = classesLoader
        Nos = self.classesNos
        fw = self.test_weight

        font = self.font
        
        def LR_graph(wlist,No):
            def node_string(count,fidx,w,No):
                nodestr = ""
                nodestr += "{} [label=< <table border=\"0\">".format(count)

                nodestr+="<tr>"
                nodestr+="<td><img src= \"{}\" /></td>".format(fpath+"{}.jpg".format(fidx)) 
                nodestr+="</tr>"


                #nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> ClassName: {} </FONT></td></tr>".format(font,classes.No2Name[No])
                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> FeatureRank: {} </FONT></td></tr>".format(font,count)

                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Feature: {}, Weight: {:.3f} </FONT></td></tr>".format(font,fidx,w)

                nodestr += "</table>  >];\n" 
                return nodestr

            resstr = "digraph Tree {node [shape=box] ;rankdir = LR;\n"


            count = len(wlist)
            for k,v in wlist:
                resstr+=node_string(count,k,v,No)
                count-=1
               # print (count,k,v)
            
            resstr += "0 [label=< <table border=\"0\">" 
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> ClassName: {} </FONT></td></tr>" .format(font,classes.No2Name[No])
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> Fidelity error: {:.3f} % </FONT></td></tr>" .format(font,self.reducer_err[self.C2IDX[No]]*100)
            resstr += "<tr><td><FONT POINT-SIZE=\"{}\"> First {} features out of {} </FONT></td></tr>" .format(font,featopk,self.n_components)
            resstr += "</table>  >];\n"
            

            resstr += "}"

            return resstr

        if not os.path.exists("./Explainers/"+title+"/GE"):
            os.mkdir("./Explainers/"+title+"/GE")
                    
        print ("Generate explanations with fullset condition")

        for i,No in tqdm(enumerate(Nos)):
            wlist = [(j,fw[j][i]) for j in fw[:,i].argsort()[-featopk:]]
           
            graph = pydotplus.graph_from_dot_data(LR_graph(wlist,No)) 
           
        
            if not os.path.exists("./Explainers/"+title+"/GE/{}.jpg".format(No)):
                graph.write_jpg("./Explainers/"+title+"/GE/{}.jpg".format(No))
                        
    def feature_UI(self,heatmapUI = False):
        utils = self.utils

        def view_img(threshold,featureNo):
            samples,heatmap = self.features[featureNo]
            nheatmap = heatmap * (heatmap>=threshold)
            nheatmap = utils.resize_img(nheatmap)
            nsamples = samples * np.repeat(nheatmap,self.nchannels).reshape(list(nheatmap.shape)+[-1])
            utils.show_img(nsamples,1,self.topk)
        return interact(view_img, threshold = (0.0,1.0,0.05), featureNo = list(self.features.keys()))


    def linear_local_explanations(self,x,model,classesLoader,target_classes = None,background = 0.2,name = None,with_total = True,display_value = True):
        utils = self.utils
        font = self.font
        featuretopk = min(self.featuretopk,self.n_components)


        if target_classes is None:
            target_classes = self.classesNos
        w = self.test_weight

        pred = model.predict(np.array([x]))[0][target_classes]

        if not os.path.exists("Explainers/"+self.title + "/explanations"):
            os.mkdir("Explainers/"+self.title + "/explanations")

        if not os.path.exists("Explainers/"+self.title + "/explanations/all"):
            os.mkdir("Explainers/"+self.title + "/explanations/all")

        fpath = "Explainers/"+self.title + "/explanations/{}"

        afpath = "Explainers/"+self.title + "/explanations/all/"

        if name is not None:
            if not os.path.exists(fpath.format(name)):
                os.mkdir(fpath.format(name))
            else:
                print ("Folder exists")
                return 
        else:
            count = 0
            while os.path.exists(fpath.format(count)):
                count+=1
            os.mkdir(fpath.format(count))
            name = str(count)

        fpath = fpath.format(name)+"/feature_{}.jpg"

        if self.reducer is not None:
            h = self.reducer.transform(model.get_feature(np.array([x]),self.layer_name))[0]
        else:
            h = model.get_feature(np.array([x]),self.layer_name)[0]

        feature_idx = []
        for cidx in target_classes:
            tw = w[:,self.C2IDX[cidx]]
            tw_idx = tw.argsort()[::-1][:featuretopk]
            feature_idx.append(tw_idx)
        feature_idx = list(set(np.concatenate(feature_idx).tolist()))

        for k in feature_idx:
            
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True

            x1,h1 = utils.img_filter(np.array([x]),np.array([h[:,:,k]]),background=background,minmax = minmax)
            x1 = utils.deprocessing(x1)
            x1 = x1 / x1.max()
            x1 = abs(x1)
            fig = utils.contour_img(x1[0],h1[0])
            fig.savefig(fpath.format(k)) 
            plt.close()

        fpath = os.getcwd() + "/Explainers/"+ self.title + "/feature_imgs/"
        spath = os.getcwd() + "/Explainers/"+ self.title + "/explanations/{}/".format(name)
        def node_string(fidx,score,weight):
            
            
            nodestr = ""
            nodestr += "<table border=\"0\">\n"
            nodestr+="<tr>"
            nodestr+="<td><img src= \"{}\" /></td>".format(spath+"feature_{}.jpg".format(fidx)) 
            nodestr+="<td><img src= \"{}\" /></td>".format(fpath+"{}.jpg".format(fidx)) 
            nodestr+="</tr>\n"
            if display_value:
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> ClassName: {}, Feature: {}</FONT></td></tr>\n".format(font,classesLoader.No2Name[cidx],fidx)
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> Similarity: {:.3f}, Weight: {:.3f}, Contribution: {:.3f}</FONT></td></tr> \n".format(font,score,weight,score*weight)
            nodestr += "</table>  \n" 
            return nodestr


        s = h.mean(axis = (0,1))
        print("s shape",s.shape )
        print("s", s)
        for i,cidx in enumerate(target_classes):
            tw = w[:,self.C2IDX[cidx]]
            tw_idx = tw.argsort()[::-1][:featuretopk] 
            
            total = 0

            resstr = "digraph Tree {node [shape=plaintext] ;\n"
            resstr += "1 [label=< \n<table border=\"0\"> \n"
            for fidx in tw_idx:
                resstr+="<tr><td>\n"
                    
                resstr+=node_string(fidx,s[fidx],tw[fidx])
                total+=s[fidx]*tw[fidx]
                    
                resstr+="</td></tr>\n"

            if with_total:
                resstr +="<tr><td><FONT POINT-SIZE=\"{}\"> Total Conrtibution: {:.3f}, Prediction: {:.3f}</FONT></td></tr> \n".format(font,total,pred[i])
            resstr += "</table> \n >];\n"
            resstr += "}"

            graph = pydotplus.graph_from_dot_data(resstr)  
            graph.write_jpg(spath+"explanation_{}.jpg".format(cidx))
            graph.write_jpg(afpath+"{}_{}.jpg".format(name,cidx))
    


    ####### Generate Our TreeICE Explanations
    def tree_model(self,x, model, classesLoader, target_classes=None, background = 0.2, name = None):
        
        featuretopk = min(self.featuretopk, self.n_components)
        utils = self.utils
        font = self.font
        random_seed = 0
        classes = [i for i,v in enumerate(classesLoader.classes)]
      
       

        if target_classes is None:
            target_classes = self.classesNos
        w = self.test_weight

        
        if not os.path.exists("./Explainers/"+self.title + "/tree/"):
            os.mkdir("./Explainers/"+self.title + "/tree/")

        if not os.path.exists("./Explainers/"+self.title + "/tree/all/"):
            os.mkdir("./Explainers/"+self.title + "/tree/all/")

        fpath = "./Explainers/"+self.title + "/tree/{}/"

        afpath = "./Explainers/"+self.title + "/tree/all/"

        if name is not None:
            if not os.path.exists(fpath.format(name)):
                os.mkdir(fpath.format(name))
            else:
                print ("Folder exists")
                return 
        else:
            count = 0
            while os.path.exists(fpath.format(count)):
                count+=1
            os.mkdir(fpath.format(count))
            name = str(count)

        fpath = fpath.format(name)+"/feature_{}.jpg"
        
        X_feature =[]
        fm = []
       
        y_true = []
        X_train = []
        
       
        for No,X,y in tqdm(classesLoader.load_train(self.classesNos)):
            X_train.append(X)

            if self.reducer is not None:
                featureMaps = self.reducer.transform(model.get_feature(X,self.layer_name))
                
            else:
                featureMaps = model.get_feature(X,self.layer_name)       
            g = self.feature_filter(featureMaps)
            X_feature.append(g)
           
            f = model.get_feature(X, self.layer_name)
            fm.append(f)
            y_true.append(y)
                    
        y_model  =  []
        y_true_f = np.hstack(y_true)
        X_feature_f = np.vstack(X_feature)
        X_f = np.vstack(X_train)
        fm_f = np.vstack(fm)
        #print("len X_f ", len (X_f))
        #print("y_true len", len(y_true_f))
        #print("len X_f ", len (fm_f))
        for i in X_f:
            pred_X = model.predict(np.array([i]))[0][target_classes]
            #print("pred_X", pred_X)
            #print("len pred_X", len(pred_X))
          
            y_model.append(target_classes[np.argmax(pred_X)])
            #print("y model " , y_model)
            #print("len y model" , len(y_model))
        

        print('Training concept tree :') 
        
        train_x, val_x, train_y, val_y = train_test_split(X_feature_f, y_model, test_size=0.3, random_state=random_seed)
        #param_grid = {'criterion':['gini','entropy'], 'max_depth' :[3,5,7,20]}
       
        surrogate_tree = DecisionTreeClassifier(max_depth = 10, random_state=random_seed)
        #grid_search = GridSearchCV(surrogate_tree,param_grid=param_grid,cv=5)
        surrogate_tree.fit(train_x, train_y)
        print("train_x", train_x)
        
        print("train_y" ,len(train_y))
        train_score = surrogate_tree.score(train_x, train_y)
        #cross_val_score(surrogate_tree, train_x, train_y, cv=10) 

        #
        test_score = surrogate_tree.score(val_x, val_y)
        print(f'train accuracy:\t {train_score}')
        print(f'test accuracy:\t {test_score}')
        
        tree_pred = surrogate_tree.predict(X_feature_f)
        print("tree_pred", tree_pred)
    

        #the prediction prob of any input image x from the pre-trained cnn model
        pred_x = model.predict(np.array([x]))[0][target_classes]
        print("pred_x", pred_x)
               
        
        if self.reducer is not None:
            h = self.reducer.transform(model.get_feature(np.array([x]),self.layer_name))[0][0]
            
        else:
            h = model.get_feature(np.array([x]),self.layer_name)[0][0]
        s = h.mean(axis = (0,1))
        

        tree_pred_x = surrogate_tree.predict(h)[0]
        print("tree_pred x", tree_pred_x)
        
        
        def plot_decision_tree(clf,feature_name,target_name):
          
           dot_data = tree.export_graphviz(clf,  
                                 feature_names=feature_name,  
                                 class_names=target_name,  
                                 filled=True, rounded=True,  
                                 special_characters=True)  
           graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
           return Image(graph.create_png())
        
           # plot_decision_tree(surrogate_tree, train_x,train_y)
       
     
    def generate_image_tree_file(self, x, model, classesLoader, name = None, classDisplay = 3):
        title = self.title   
        imgtopk = self.featureimgtopk
        classes = classesLoader
        Nos = self.classesNos
        font = self.font
        
        if not os.path.exists("./Explainers/"+self.title + "/tree/"):
            os.mkdir("./Explainers/"+self.title + "/tree/")

        if not os.path.exists("./Explainers/"+self.title + "/tree/all/"):
            os.mkdir("./Explainers/"+self.title + "/tree/all/")

        fpath = "./Explainers/"+self.title + "/tree/{}/"

        afpath = "./Explainers/"+self.title + "/tree/all/"

        if name is not None:
            if not os.path.exists(fpath.format(name)):
                os.mkdir(fpath.format(name))
            else:
                print ("Folder exists")
                return 
        else:
            count = 0
            while os.path.exists(fpath.format(count)):
                count+=1
            os.mkdir(fpath.format(count))
            name = str(count)

        fpath = fpath.format(name)+"/feature_{}.jpg"
        
        fpath = os.getcwd() + "/Explainers/"+ self.title + "/feature_imgs/"
        spath = os.getcwd() + "/Explainers/"+ self.title + "/tree/{}/".format(name)
        
        fv = []
        y_model = [] 
        X_img = []
        
        X_feature = []
        y_true = []
        for No, X,y in tqdm(classesLoader.load_train(self.classesNos)):
                    featureMaps = self.reducer.transform(model.get_feature(X,self.layer_name))
                   # print("featureMaps", featureMaps.shape)
                    X_fvector = self.feature_filter(featureMaps) 
                    #featureMaps.mean(axis = (0,1))
                    print("X_fvector", X_fvector.shape)
                    print("X_fvector [0]", X_fvector[0].shape)
                    fv.append(X_fvector)
                    
                    X_fm = model.get_feature(X,self.layer_name)
                    X_feature.append(X_fm)
                    X_img.append(X)
                    y_true.append(y)
           
                    for i, x in enumerate(X):
                        pred_X = model.predict(np.array([x]))[0][Nos]
                        y_model.append(Nos[np.argmax(pred_X)])

            
        concept_vectors = np.vstack(fv)
        y_true_f = np.hstack(y_true)

        print('Training concept tree :') 
        param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(12, 25) }# 'max_leaf_nodes': np.arange(2, 10) }#'min_samples_split': np.arange(5, 10)}
        grid_search_cv = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid, cv = 10)
        grid_search_cv.fit(concept_vectors,y_model)    
        print("best_tree_params", grid_search_cv.best_params_)
        
        self.tree_model =  grid_search_cv.best_estimator_  
        #train_x, val_x, train_y, val_y = train_test_split(concept_vectors, y_model, test_size=0.3)

        self.tree_model.fit(concept_vectors, y_model)
                              
        train_score = self.tree_model.score(concept_vectors, y_model)
        print(f'Average Fidelity score of the target classes on the training set:\t {train_score}')
        
        ##### Evaluation #######
        #create validation dataset 
        fv_val = []
        y_val =  []
        y_model_val = []
        X_img_val = []
        for no, inst,label in tqdm(classesLoader.load_val(self.classesNos)):
                    featureMaps_v = self.reducer.transform(model.get_feature(inst,self.layer_name))
                    X_fvector_val = self.feature_filter(featureMaps_v) 
                    fv_val.append(X_fvector_val)
                    
                    X_img_val.append(inst)
                    print("inst", len(inst))
                    y_val.append(label)
           
                    for i, s in enumerate(inst):
                        pred_X_val = model.predict(np.array([s]))[0][Nos]
                        y_model_val.append(Nos[np.argmax(pred_X_val)])
            
        concepts_val = np.vstack(fv_val)
        y_val_f = np.hstack(y_val)
        
        fid_score_all = self.tree_model.score(concepts_val, y_model_val)
        print(f' Average Fidelity score of the target classes on the validation set with the CNN model predictions):\t {fid_score_all}')
        
        test_score = self.tree_model.score(concepts_val, y_val_f)
        print(f' Accuracy of the Decision Tree Model on the validation set with the ground-truth predictions):\t {test_score}')
                         
        tree  = self.tree_model.tree_
        print(" tree.f",len(tree.feature))
        print(" tree.f", tree.feature)
        
        train_f1score = metrics.f1_score(y_model, self.tree_model.predict(concept_vectors), average='weighted')
        test_f1score = metrics.f1_score(y_model_val, self.tree_model.predict(concepts_val), average='weighted')
        print(f'Train F1 Score of the Decision Tree Model with the CNN model predictions:\t{train_f1score}')
        print(f'Test F1 Score of the Decision Tree Model with the CNN model predictions:\t {test_f1score}')
        
        c_f1score = metrics.f1_score(y_true_f, self.tree_model.predict(concept_vectors), average='weighted')
        print(f' Train F1 Score ofthe Decision Tree Model with the ground-truth labels:\t{c_f1score}')
     
        c_f1score_val = metrics.f1_score(y_val_f, self.tree_model.predict(concepts_val), average='weighted')
        print(f'F1 Score of the Decision Tree Model with the ground-truth labels on validation set:\t{c_f1score_val}')
        
        #fidelity score  for each target class
        fid = []
        for i in range(len(Nos)):
            p_x = model.predict((X_img_val[i]))[:,Nos]    
            m1 = [Nos[np.argmax(subarray)] for subarray in p_x]
            tree_res = self.tree_model.predict(fv_val[i])
            matches = ( m1 == tree_res)
            fid.append(matches.sum()  / float(len(matches)))

        tree_fid = np.array(fid)

        if type(tree_fid) is not np.ndarray:
            tree_fid = np.array([tree_fid])

        print ("Fidelity score of the Decision Tree Model is: {}".format(tree_fid))
                
        ####### Decision Tree Model visualization #######
        resstr = "digraph Tree {node [shape=box] ;\n"
        def node_string(node):
            nodestr = ""
            nodestr += "{} [label=< <table border=\"0\">".format(node)
            
            if tree.feature[node]>=0:
                nodestr+="<tr>"
                nodestr+="<td><img src= \"{}\" /></td>".format(fpath+"{}.jpg".format(tree.feature[node])) # fixedsize=\"true\" width=\"50%\"
                nodestr+="</tr>"
                
                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Feature: {}, Threshold: {:.3f} </FONT></td></tr>".format(font,tree.feature[node],tree.threshold[node],tree.n_node_samples[node])
            nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Num_Samples: {} </FONT></td></tr>".format(font,tree.n_node_samples[node])
            
            prob = tree.value[node][0]
            prob = prob / prob.sum()
            pidx = prob.argsort()[::-1]
           

            for cidx in range(min(classDisplay,len(self.classesNos))):        
                    nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Class: {}, Proportion: {:.2f}, Fidelity_Score: {:.3f} </FONT></td></tr> \n".format(font,classes.No2Name[Nos[pidx[cidx]]], prob[pidx[cidx]], tree_fid[cidx])
                #    nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Fidelity_Error: {:.3f} % </FONT></td></tr> \n" .format(font, self.reducer_err[self.C2IDX[cidx]]*100)
            nodestr += "</table>  >];\n" 
            return nodestr
        
          
        for i in range(len(tree.feature)):
                resstr += node_string(i)
            
        for i in range(len(tree.feature)):
                lc = tree.children_left[i]
                rc = tree.children_right[i]
                
                if lc>=0:
                    resstr += "{} -> {} [labeldistance=2.5, labelangle=100,fontsize={} headlabel=\"{}\"] ;\n".format(i,lc,font,"Not Similar / Smaller")#,"Not Similar / Smaller")
                    
                if rc>=0:
                    resstr += "{} -> {} [labeldistance=2.5, labelangle=-100,fontsize={} headlabel=\"{}\"] ;\n".format(i,rc,font,"Similar / Greater")#,"Similar / Greater")
           
        resstr += "}" 
    
        graph = pydotplus.graph_from_dot_data(resstr)
        if not os.path.exists("./Explainers/"+title+ "Tree_VIS.jpg"):
                graph.write_jpg(spath+"Tree_VIS.jpg")
        
                
    
    def tree_local(self,x,model,classesLoader,target_classes = None,background = 0.2,name = None,with_total = True,display_value = True, classDisplay = 3):
        utils = self.utils
        title = self.title   
        imgtopk = self.featureimgtopk
        classes = classesLoader
        Nos = self.classesNos
        font = self.font
        featuretopk = min(self.featuretopk,self.n_components)
        
        if target_classes is None:
            target_classes = self.classesNos
        w = self.test_weight
                
        if not os.path.exists("Explainers/"+self.title + "/tree_local_exp"):
            os.mkdir("Explainers/"+self.title + "/tree_local_exp")


        fpath = "Explainers/"+self.title + "/tree_local_exp/{}"

        if name is not None:
            if not os.path.exists(fpath.format(name)):
                os.mkdir(fpath.format(name))
            else:
                print ("Folder exists")
                return 
        else:
            count = 0
            while os.path.exists(fpath.format(count)):
                count+=1
            os.mkdir(fpath.format(count))
            name = str(count)

        fpath = fpath.format(name)+"/feature_{}.jpg"
        
        x_pred = model.predict(np.array([x]))[0][target_classes]
   
        if self.reducer is not None:
            # The reduced feature map of the new image to be explained
            h = self.reducer.transform(model.get_feature(np.array([x]),self.layer_name))

        else:
            h = model.get_feature(np.array([x]),self.layer_name)
            
        h_vector = self.feature_filter(h) # feature vectors of the reduced feature map
        #print("h_feature", h_vector)
        #print("h_feature", h_vector.shape)

        feature_idx = []
        for cidx in target_classes:
            tw = w[:,self.C2IDX[cidx]]
            tw_idx = tw.argsort()[::-1][:featuretopk]
            feature_idx.append(tw_idx)
        feature_idx = list(set(np.concatenate(feature_idx).tolist()))
    
        t = h[0]

        for k in feature_idx:     
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True

            x1,h1 = utils.img_filter(np.array([x]),np.array([t[:,:,k]]),background=background,minmax = minmax)
            x1 = utils.deprocessing(x1)
            x1 = x1 / x1.max()
            x1 = abs(x1)
            fig = utils.contour_img(x1[0],h1[0])
            fig.savefig(fpath.format(k)) 
            plt.close()
     
        fpath = os.getcwd() + "/Explainers/"+ self.title + "/feature_imgs/"
        spath = os.getcwd() + "/Explainers/"+ self.title + "/tree_local_exp/{}/".format(name)
        
        h_pred_class = self.tree_model.predict(h_vector)  
        print("The new image most likely belongs to the class:", classes.No2Name[Nos[Nos.index(h_pred_class[0])]] )
        
        input_score = h[0].mean(axis = (0,1)) # features(concepts) scores of a new image input to be explained
        tree  = self.tree_model.tree_
     
        def node_string(node, score):
            
            nodestr = ""
            nodestr += "{} [label=< <table border=\"0\">".format(node)
            
            if tree.feature[node]>=0:  
                nodestr+="<tr>"
                nodestr+="<td><img src= \"{}\" /></td>".format(spath+"feature_{}.jpg".format(tree.feature[node]))
                nodestr+="<td><img src= \"{}\" /></td>".format(fpath + "{}.jpg".format(tree.feature[node])) 
                nodestr+="</tr>\n"
                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Concept Feature: {}, Threshold: {:.3f} </FONT></td></tr>".format(font,tree.feature[node],tree.threshold[node])
                nodestr +="<tr><td colspan=\"2\"><FONT POINT-SIZE=\"{}\"> Similarity: {:.3f}</FONT></td></tr> \n".format(font, input_score[tree.feature[node]],tree.feature[node])
                       
          
          
            prob = tree.value[node][0]           
            prob = prob / prob.sum()
            pidx = prob.argsort()[::-1]  
            
            
            
            for cidx in range(min(classDisplay,len(self.classesNos))):
                nodestr +="<tr><td><FONT POINT-SIZE=\"{}\"> Class: {}, Proportion: {:.2f}</FONT></td></tr> \n".format(font,classes.No2Name[Nos[pidx[cidx]]], prob[pidx[cidx]])    
                
                
                
            nodestr += "</table>  >];\n" 
        
    
     
            return nodestr
            
        resstr = "digraph Tree {node [shape=box] ;\n"              
        
        for i in range(len(tree.feature)):
             
                resstr+= node_string(i, s[i])
                                 
        for i in range(len(tree.feature)):
                            lc = tree.children_left[i]
                            rc = tree.children_right[i]     
                            if lc>=0:
                                resstr += "{} -> {} [labeldistance=2.5, labelangle=100,fontsize={} headlabel=\"{}\"] ;\n".format(i,lc,font,"Not Similar / Smaller")#,"Not Similar / Smaller")
                                
                            if rc>=0:
                                resstr += "{} -> {} [labeldistance=2.5, labelangle=-100,fontsize={} headlabel=\"{}\"] ;\n".format(i,rc,font,"Similar / Greater")#,"Similar / Greater")
        
        
        decision_paths = self.tree_model.decision_path(h_vector)
        #The leaf ids reached by samples of interest can be obtained with the apply method
        leaf_id = self.tree_model.apply(h_vector)
        print("leaf_id",leaf_id)
 
        sample_id = 0
        
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = decision_paths.indices[decision_paths.indptr[sample_id]:
                                    decision_paths.indptr[sample_id + 1]]


        for node_id in node_index:
            # continue to the next node if it is a leaf node
                if leaf_id[sample_id] == node_id:
                    continue
        
        
        print('Rules used to predict sample {id}:\n'.format(id=sample_id))
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue
            

             # check if value of the split feature for sample 0 is below threshold
            if (h_vector[sample_id, tree.feature[node_id]] <= tree.threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

        print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
              "{inequality} {threshold})".format(
                  node=node_id,
                  sample=sample_id,
                  feature=tree.feature[node_id],
                  value=h_vector[sample_id, tree.feature[node_id]],
                  inequality=threshold_sign,
                  threshold=tree.threshold[node_id]))
 
        
        
        resstr += "}"
              
        graph = pydotplus.graph_from_dot_data(resstr)  


        
        graph.write_jpg(spath+"tree_local_exp_{}.jpg".format(name))
        graph.write_jpg(fpath+"{}_{}.jpg")