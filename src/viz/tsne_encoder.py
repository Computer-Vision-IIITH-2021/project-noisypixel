import torch
import numpy as np
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                         shuffle=False, num_workers=2)


model = models.resnet101(pretrained=True)
model_weights = [] 
conv_layers = [] 
model_children = list(model.children())


counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")


for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

print(data.shape)
results = [conv_layers[0](data)]
print(results[0].shape)
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
    print(results[-1].shape)
# make a copy of the `results`plt.scatter(ty,[i for i in range(len(ty))])
# outputs = results[-1]#.view(results[-1].shape[0],-1)
# print(outputs.shape)
m = nn.AvgPool2d(2, stride=2)
output = m(results[-1])
outputs = output.view(output.shape[0],-1)
print(outputs.shape)
out = outputs.cpu().detach().numpy()


kmeans = KMeans(n_clusters=10, random_state=0).fit(out)
lab = kmeans.predict(out)

tsne = TSNE(n_components=2).fit_transform(out)