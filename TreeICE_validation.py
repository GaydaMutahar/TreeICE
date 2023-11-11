'''
@Author: Gayda Mutahar
@Date: 2021-10-31
@LastEditors  : Gayda Mutahar
@LastEditTime : 2023-11-01
@Description: This script to perform the entire computional evaluation of TreeICE framework
'''
import torchvision.models as models
import torch
import torch.nn as nn



import utils
from Explainer import *
import Explainer
import importlib
import numpy as np
import utils
import time as time 


start = time.time()

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
use_cuda = torch.cuda.is_available() #to verify that cuda is supported

device = torch.device('cuda' if use_cuda else 'cpu')
save_dir = '/Users/ghaidaa/Desktop/InvertibleCE-main/models/resnet50.pkl'

m = m.load_state_dict(torch.load(save_dir), map_location=device)


#m = m.cuda()
m.eval()
model = PytorchModelWrapper(m,batch_size=64,input_size = [3,448,448],layer_dict = dict(m.module.base_model.named_children()))


from torchvision import transforms
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
classes = CUBBirdClassesLoader(path = '/Users/ghaidaa/Desktop/InvertibleCE-main/dataset/CUB_200_2011/data',label_path = "classes.txt", preprocess_input=transform,target_size = [448,448])

layer_dict = dict(m.module.base_model.named_children())
layer_dict

class_labels=[]
original_model_acc = []
linear_model_acc = []
tree_model_acc = []
linear_model_fid = []
tree_model_fid =  []
original_model_f1score = []
linear_model_acc_f1score = []
tree_model_acc_f1score = []
linear_model_fid_f1score = []
tree_model_fid_f1score = []



# Averages
average_original_model_acc = []
average_linear_model_acc = []
average_tree_model_acc = []
average_linear_model_fid = []
average_tree_model_fid =  []

average_original_model_acc_f1score = []
average_linear_model_acc_f1score = []
average_tree_model_acc_f1score = []
average_linear_model_fid_f1score = []
average_tree_model_fid_f1score = []

for i in range(200):
  class_labels.append(i)

number_classes = [4,6,8,10,12,14]
number_components = [5,10,15,20,25,30]


print("############################")
print("############################")
#print("iteration #", iteration)
for n_class in number_classes:
  print("Number of classes in this trail is", n_class)
  ran_classes = random.sample(class_labels, n_class)
  title = "classes:" + str(ran_classes)
  print(title)

  # target layer name
  layer_name = "layer4"
  # set predict layer and target classes
  model.target = ("fc",np.array(ran_classes))
  classes.set_target(ran_classes)
  print(ran_classes)
  
  for n_component in number_components:
    Exp = Explainer.Explainer(title = title ,
                layer_name = layer_name,
                classesNos = ran_classes,
                n_components = n_component,
                utils = utils.utils(mode = "torch",img_size = (448,448,3)),
                reducer_type = "NMF"
              )
    print("Number of concepts (n_component) in this trail is: ", n_component)
        # train reducer based on target classes
    try:
        Exp.train_model(model,classes)
    except:
        Exp.reducer = None
        Exp.train_model(model,classes) 
    # generate features 
    #Exp.generate_features(model,classes)
    # save features with visualizations
   # Exp.save_features()
    # generate global linear explanations
    #Exp.LR_global_explanation(classes)
    # save the explainer, use load to load it with the same title
    Exp.save()
    # generate a tree decision tree model for global explanation 
    Exp.tree_global_explantion(model, classes)  
    # evaluate both linear and tree explanations 
    eval = Exp.computational_eval(model,classes)
    print("evaluation values:", eval)
    original_model_acc.append(eval[0])
    linear_model_acc.append(eval[1])
    tree_model_acc.append(eval[2])
    linear_model_fid.append(eval[3]) 
    tree_model_fid.append(eval[4])
    original_model_f1score.append(eval[5])
    linear_model_acc_f1score.append(eval[6])
    tree_model_acc_f1score.append(eval[7])
    linear_model_fid_f1score.append(eval[8])
    tree_model_fid_f1score.append(eval[9])  

  
print("############################")  
print("############################")
print("############################")
print("list of Accuracy scores of the original model, using accuracy score", original_model_acc)
print("############################")
print("list of Accuracy scores of the linear model, using accuracy score:", linear_model_acc)
print("############################")
print("list of Accuracy scores of the Tree model, using accuracy score:", tree_model_acc)
print("############################")
print("############################")
print("############################")

print("list of Fidelity scores of the linear model, using accuracy score", linear_model_fid )
print("############################")
print("list of Fidelity scores of the Tree model, using accuracy score:",tree_model_fid )
print("############################")
print("############################")
print("############################")

print("list of Accuracy scores of the original model, using F1 score", original_model_f1score)
print("############################")
print("list of Accuracy scores of the linear model, using F1 score:", linear_model_acc_f1score)
print("############################")
print("list of Accuracy scores of the Tree model, using F1 score:", tree_model_acc_f1score)
print("############################")
print("############################")
print("############################")

print("list of Fidelity scores of the linear model, using F1 score:", linear_model_fid_f1score )
print("list of Fidelity scores of the Tree model,using F1 score:", tree_model_fid_f1score )
print("############################")
print("############################")
print("############################")

end = time.time()
print(f"Execution time is :", {end - start})


#Visulisation of the Computional evaluation using annotated heatmap

# LR Model ACCURACY

# Create a 2D grid for the heatmap
acc_matrix = np.array(linear_model_acc_f1score).reshape(len(number_classes), len(number_components))


# Create the heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size


# Loop through rows and columns to annotate each cell with accuracy values
for i in range(len(number_classes)):
    for j in range(len(number_components)):
        plt.text(j + 0.5, i + 0.5, f'{acc_matrix[i, j]:.2f}', ha='center', va='center', color='grey', fontsize=12, fontweight='bold')

heatmap = sns.heatmap(acc_matrix, cmap='YlGnBu', xticklabels=number_components, yticklabels=number_classes, cbar=True)

plt.xlabel('NCAVs number', fontsize=12)
plt.ylabel('Classes number', fontsize=12)
#plt.title('Accuracy Heatmap for LR Model, using F1 Score', fontsize=15, fontweight='bold')
plt.yticks(rotation=20) 
plt.show()
#plt.savefig('linear_model_acc.png')


#######################################################
# DT model ACCURACY
# Create a 2D grid for the heatmap
tree_model_acc_matrix = np.array(tree_model_acc_f1score).reshape(len(number_lasses), len(number_components))


# Create the heatmap using Seaborn
plt.figure(figsize=(8, 6))
#sns.set(font_scale=1.1)  # Adjust font size
# Define a custom colormap (you can replace this with any colormap you prefer)
custom_cmap = sns.color_palette("coolwarm", as_cmap=True)


# Loop through rows and columns to annotate each cell with accuracy values
for i in range(len(number_classes)):
    for j in range(len(number_components)):
        plt.text(j + 0.5, i + 0.5, f'{tree_model_acc_matrix[i, j]:.2f}', ha='center', va='center', color='w', fontsize=12, style='normal', fontweight='bold')

heatmap = sns.heatmap(tree_model_acc_matrix, cmap=custom_cmap, xticklabels=number_components, yticklabels=number_classes, cbar=True)


plt.xlabel('NCAVs number' , fontsize=12)
plt.ylabel('Classes number', fontsize=12)
plt.title('Accuracy Heatmap for DT Model, using F1 Score', fontsize=15, fontweight='bold')
plt.yticks(rotation=20)  # Rotate by 0 degrees
plt.savefig('tree_model_acc.png')

plt.show()


##########################################
#LR Fidelity 
# Create a 2D grid for the heatmap
linear_model_fid_matrix = np.array(linear_model_fid_f1score).reshape(len(number_classes), len(v))


# Create the heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size
# Define a custom colormap (you can replace this with any colormap you prefer)
custom_cmap = sns.color_palette("YlGnBu", as_cmap=True)


# Loop through rows and columns to annotate each cell with accuracy values
for i in range(len(number_classes)):
    for j in range(len(number_components)):
        plt.text(j + 0.5, i + 0.5, f'{linear_model_fid_matrix[i, j]:.2f}', ha='center', va='center', color='grey', fontsize=12, style='normal', fontweight='bold')

heatmap = sns.heatmap(linear_model_fid_matrix, cmap=custom_cmap, xticklabels=v, yticklabels=number_classes, cbar=True)


plt.xlabel('NCAVs number', fontsize=12)
plt.ylabel('Classes number', fontsize=12)
plt.title('Fidelity Heatmap for LR Model, using F1 Score', fontsize=15, fontweight='bold')
plt.yticks(rotation=20)  # Rotate by 0 degrees
#plt.savefig('linear_model_fid.png')


plt.show()
################################################

# Create a 2D grid for the heatmap
tree_model_fid_matrix = np.array(tree_model_fid_f1score).reshape(len(number_classes), len(number_components))


# Create the heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size
# Define a custom colormap (you can replace this with any colormap you prefer)
custom_cmap = sns.color_palette("coolwarm", as_cmap=True)


# Loop through rows and columns to annotate each cell with accuracy values
for i in range(len(number_classes)):
    for j in range(len(number_components)):
        plt.text(j + 0.5, i + 0.5, f'{tree_model_fid_matrix[i, j]:.2f}', ha='center', va='center', color='w', fontsize=12, style='normal', fontweight='bold')

heatmap = sns.heatmap(tree_model_fid_matrix, cmap=custom_cmap, xticklabels=number_components, yticklabels=number_of_classes, cbar=True)


plt.xlabel('NCAVs number', fontsize=12)
plt.ylabel('Classes number', fontsize=12)
plt.title('Fidelity Heatmap for DT Model, using F1 Score', fontsize=15, fontweight='bold')
plt.yticks(rotation=20)  # Rotate by 0 degrees
#plt.savefig('tree_model_fid.png')

plt.show()

"""
###################averges#################
average_original_model_acc.append(original_model_acc)
  average_linear_model_acc.append(linear_model_acc)
  average_tree_model_acc.append(tree_model_acc)
  

  average_linear_model_fid.append(linear_model_fid)
  average_tree_model_fid.append(tree_model_fid)

  average_original_model_acc_f1score.append(original_model_f1score)
  average_linear_model_acc_f1score.append(linear_model_acc_f1score)
  average_tree_model_acc_f1score.append(tree_model_acc_f1score)

  average_linear_model_fid_f1score.append(linear_model_fid_f1score)
  average_tree_model_fid_f1score.append(tree_model_fid_f1score)
print("###################averges#################")
print("average_original_model_acc", average_original_model_acc)
print("average_linear_model_acc", average_linear_model_acc)
print("average_tree_model_acc", average_tree_model_acc)
print("########")
print("############################")
print("average_linear_model_fid", average_linear_model_fid)
print("average_tree_model_fid", average_tree_model_fid)
print("########")
print("############################")

print("average_original_model_acc_f1score", average_original_model_acc_f1score)
print("average_linear_model_acc_f1score", average_linear_model_acc_f1score)
print("average_tree_model_acc_f1score", average_tree_model_acc_f1score)
print("########")
print("############################")
print("average_linear_model_fid_f1score", average_linear_model_fid_f1score)
print("average_tree_model_fid_f1score", average_tree_model_fid_f1score)


end = time.time()
print(f"Execution time is :", {end - start})

#############################################

#Visulisation of the Computional evaluation  
#plot the accuracy of the three models, the original model and the two explantion models

print("Plotting bar chart of the accuracy of the three models")
data_accu = [original_model_acc, linear_model_acc, tree_model_acc]
plot1 = plt.figure(1,figsize=(55,12))
x = list()
step = 0.7/(len(data_accu)+1)
for i in range(len(data_accu)):
  x.append([j+step*i for j in range(len(data_accu[i]))])
colors = ["tab:orange","tab:green","tab:blue"]
labels = ['Original Model','Linear Model', 'Decision Tree Model']
for i in range(len(data_accu)):
  plt.bar(x[i], data_accu[i], color=colors[i], width=step, label=labels[i])
x_labels = list()
for c in number_classes:
  for f in number_components:
    x_labels.append(f'c={c}, f={f}')

plt.title('Accuracy')
plt.xticks(np.arange(0, len(data_accu[0]), step=1), x_labels)
plt.legend(loc='lower right')#bbox_to_anchor=(1.05, 1))#loc='lower left)
plt.savefig('accuracy_bar_chart.png')


print("Plotting line chart of the accuracy of the three models")
plot2 = plt.figure(2, figsize=(55,12))
x_val = [i for i in range(len(data_accu[0]))]
for i in range(len(data_accu)):
  plt.plot(x_val, data_accu[i], color=colors[i], label=labels[i], marker='o' )

plt.title('Accuracy')
plt.xticks(np.arange(0, len(data_accu[0]), step=1), x_labels)
plt.legend()
plt.savefig('accuracy_line_chart.png')


###############################
#plot the fidelity of the both explantion models, along with original model labels
print("Plotting bar chart of the fidelity for the two explantion models")
data_fid = [linear_model_fid, tree_model_fid]
plot3 = plt.figure(3, figsize=(55,12))
x1 = list()
step1 = 0.7/(len(data_fid)+1)
for i in range(len(data_fid)):
  x1.append([j+step*i for j in range(len(data_fid[i]))])
colors1 = ["tab:orange","tab:blue"]
labels1 = ['Linear Model', 'Decision Tree Model']
for i in range(len(data_fid)):
  plt.bar(x[i], data_fid[i], color=colors[i], width=step, label=labels[i])
x_labels1 = list()
for c1 in number_classes:
  for f1 in number_components:
    x_labels1.append(f'c={c1}, f={f1}')

plt.title('Fidelity')
plt.xticks(np.arange(0, len(data_fid[0]), step=1), x_labels1)
plt.legend(loc='lower left')
plt.savefig('Fidelity_bar_chart.png')


print("Plotting line chart of the fidelity for the two explantion models")
plot4 = plt.figure(4, figsize=(55,12))
x_val1 = [i for i in range(len(data_fid[0]))]
for i in range(len(data_fid)):
  plt.plot(x_val1, data_fid[i], color=colors1[i], label=labels1[i], marker='o' )
plt.title('Fidelity')
plt.xticks(np.arange(0, len(data_accu[0]), step=1), x_labels1)
plt.legend()
plt.savefig('Fidelity_line_chart.png')

###############################
#plot the F1 score of the both explantion models, along with original model labels
print("Plotting bar chart of the F1 score for the two explantion models")
data_fscore = [original_model_f1score, linear_model_f1score, tree_model_f1score]
plot5 = plt.figure(5,figsize=(55,12))

x3 = list()
step3 = 0.6/(len(data_fscore)+1)
for i in range(len(data_fscore)):
  x3.append([j+step3*i for j in range(len(data_fscore[i]))])
colors = ["tab:orange","tab:green","tab:blue"]
labels = ['Original Model','Linear Model', 'Decision Tree Model']
for i in range(len(data_fscore)):
  plt.bar(x3[i], data_fscore[i], color=colors[i], width=step3, label=labels[i])
x_labels = list()
for c in number_classes:
  for f in number_components:
    x_labels.append(f'{c}, {f}')

plt.title('F1 score')
plt.xticks(np.arange(0, len(data_fscore[0]), step=1), x_labels)
plt.legend(loc='lower right')#bbox_to_anchor=(1.05, 1))#loc='lower left)
plt.savefig('F1_score_bar_graph.png')


print("Plotting a line chart of the accuracy of the three models")
plot6 = plt.figure(6,figsize=(55,5))
x_val3 = [i for i in range(len(data_fscore[0]))]
for i in range(len(data_fscore)):
  plt.plot(x_val3, data_fscore[i], color=colors[i], label=labels[i] )
plt.title('F1 score')
plt.xticks(np.arange(0, len(data_fscore[0]), step=1), x_labels)
plt.legend()
plt.savefig('F1_score_line_graph.png')
plt.show()


#3D plotting
#accuracy 
data_accu = [original_model_acc, linear_model_acc, tree_model_acc]

plot7 = plt.figure(7, figsize=(10,5))
ax = plt.axes(projection='3d')

x = list()
step = 0.6/(len(data_accu)+1)

colors = ["tab:orange","tab:green","tab:blue"]
labels = ['Original Model','Linear Model', 'Decision Tree Model']
x_labels = list()
x_val = list()
y_val = list()
for c in number_classes:
  x_val.append(c)
for f in number_components:
  y_val.append(f)


x_val, y_val = np.meshgrid(x_val, y_val)

#x_val = [i for i in range(len(data_accu[0]))]
datasets = list()
for data in data_accu:
  dataset = list()
  for i in range(0, len(data), len(number_components)):
    dataset.append(data[i:i+len(number_components)])
  datasets.append(dataset)

for i in range(len(data_accu)):
  #Z = np.sin((x_val**2 + y_val**2)/2)
  #print(f'n.sin={Z}, {len(Z[0])}')
  Z = np.array(datasets[i])
  #print(f'array={Z}')
  ax.plot_wireframe(x_val, y_val, Z, edgecolor=colors[i], label=labels[i])
  #break
ax.set_title('Accuracy')
ax.set_xlabel('Classes')
ax.set_ylabel('Components')
ax.set_zlabel('Accuracy')
#ax.set_xticks(np.arange(0, 36, step=1), x_labels)
#ax.legend()
pltLabels = list()
import matplotlib as mpl
for i in range(len(labels)):
  pltLabel = mpl.lines.Line2D([0],[0], c=colors[i])
  pltLabels.append(pltLabel)

plt.legend(pltLabels, labels)
plt.show()
"""