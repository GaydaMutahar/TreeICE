'''
@Author: Gayda Mutahar
@Date: 2021-11-05 
@LastEditors: Gayda Mutahar
@LastEditTime: 2023-11-12 
@Description: To Run the framework and generate both ICE and TreeICE explainations
'''

import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms



#to verify that cuda is supported
torch.cuda.is_available()


from ClassesLoader import *
from ModelWrapper import *
import utils
from Explainer import *
import Explainer
import importlib
import numpy as np
import utils


class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(2048, n_class)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)



m = nn.DataParallel(ResNet())
save_dir = "./models/resnet50.pkl"
m.load_state_dict(torch.load(save_dir))

m = m.cuda()
m.eval()
model = PytorchModelWrapper(m,batch_size=64,input_size = [3,448,448],layer_dict = dict(m.module.base_model.named_children()))



transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
test_transforms_list = [
    transforms.ToPILImage(),
    transforms.Resize(int(448/0.875)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
]
transform = transforms.Compose(test_transforms_list)
classes = CUBBirdClassesLoader(preprocess_input=transform,target_size = [448,448])



# calculate the pre-trained model accuracy on the Validation dataset
class_labels=[]
for i in range(200):
    class_labels.append(i)
    
y_model = [] 
y_ = []

for (n,X,y) in (classes.load_val(class_labels)):
                    y_.append(y)
                    for i, x in enumerate(X):
                        pred_X = model.predict(np.array([x]))[0][class_labels]
                        y_model.append(class_labels[np.argmax(pred_X)])
y_true = np.hstack(y_)
print("y_true", y_true.shape)
print("y_true", y_true)

print("y_model ", y_model)
print("y_model ", len(y_model))

print("The model accuracy on validation data" , metrics.accuracy_score(y_true,y_model)*100,"%")

print("The model F1 score on validation data" , metrics.f1_score(y_true, y_model, average='weighted')*100,"%")



# set target classes
target_class = [5, 10, 50, 70]
title = "cubbirds_all"

# target layer name
layer_name = "layer4"
# set predict layer and target classes
model.target = ("fc",np.array(target_class))
classes.set_target(target_class)


# create an Explainer
Exp = Explainer.Explainer(title = title,
                layer_name = layer_name,
                classesNos = target_class,
                n_components = 15,
                utils = utils.utils(mode = "torch",img_size = (448,448,3)),
                reducer_type = "NMF"
               )



# train reducer based on target classes
try:
    Exp.train_model(model,target_class)
except:
    Exp.reducer = None
    Exp.train_model(model,target_class) 
# generate features 
Exp.generate_features(model,target_class)
# save features with visualizations
Exp.save_features()
# generate global explanations
Exp.generate_image_LR_file(target_class)
# save the explainer, use load to load it with the same title
Exp.save()


#Loed the explainer
Exp.load()



# generate linear local explanation for bird.png  (Local ICE)
from PIL import Image
img_size = [448,448]
img_path = "r_bird.jpeg"
img = Image.open(img_path).convert("RGB")
input_image = transforms.ToTensor()(transforms.Resize(img_size)(img))
input_image = classes.preprocess_input(input_image).numpy()
input_image = np.transpose(input_image,(1,2,0)) #to channel_last
y = Exp.linear_local_explanations(input_image,model,target_class)




# global tree explanation (Global TreeICE)
Exp.generate_image_tree_file(input_image, model, target_class)



# generate tree local explanation for bird.png (Local TreeICE)
Exp.tree_local(input_image , model, class_labels)


# The prediction of the original CNN model for the new image x 
pred_x = model.predict(np.array([input_image]))[0][target_class]
pred_x
y_model = []
y_model.append(target_class[np.argmax(pred_x)])
print("The prediction of the original CNN model for the new image x :", classes.No2Name[target_class[target_class.index(y_model[0])]])
