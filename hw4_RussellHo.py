#Import the necessary modules for the job
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
from torchvision import utils
from torch.utils.data import DataLoader
import os
from hw4_dataloader import MyDataset
from cnn_net1 import HW4Net
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F

#Creating Net2 class that inherits from HW4Net
class HW4Net2(HW4Net):
    def __init__( self ):
        super (HW4Net , self ). __init__ ()
        self . conv1 = nn. Conv2d (3, 16 , 3, padding = 1)
        self . pool = nn. MaxPool2d (2, 2)
        self . conv2 = nn. Conv2d (16 , 32 , 3, padding = 1)
        self . fc1 = nn. Linear (8192 , 64) # output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
        self . fc2 = nn. Linear (64 , 5)    #The number is 5 because the classes are [airplane, bus, cat, dog, pizza]

#Creating Net3 class that inherits from HW4Net as well
class HW4Net3(HW4Net):
    def __init__( self ):
        super (HW4Net , self ). __init__ ()
        self . conv1 = nn. Conv2d (3, 16 , 3, padding = 1)
        self . pool = nn. MaxPool2d (2, 2)
        self . conv2 = nn. Conv2d (16 , 32 , 3, padding = 1)
        #Extra Ten Layers
        self . conv3 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv4 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv5 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv6 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv7 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv8 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv9 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv10 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv11 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . conv12 = nn. Conv2d (32 , 32 , 3, padding = 1)
        self . fc1 = nn. Linear (8192, 64) # output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
        self . fc2 = nn. Linear (64 , 5)    #The number is 5 because the classes are [airplane, bus, cat, dog, pizza]
    #Forward Method
    def forward (self , x):
        x = self . pool (F. relu ( self . conv1 (x)))
        x = self . pool (F. relu ( self . conv2 (x)))
        x = F. relu ( self . conv3 (x))
        x = F. relu ( self . conv4 (x))
        x = F. relu ( self . conv5 (x))
        x = F. relu ( self . conv6 (x))
        x = F. relu ( self . conv7 (x))
        x = F. relu ( self . conv8 (x))
        x = F. relu ( self . conv9 (x))
        x = F. relu ( self . conv10 (x))
        x = F. relu ( self . conv11 (x))
        x = F. relu ( self . conv12 (x))
        x = x. view (x.shape[0], -1)
        x = F. relu ( self .fc1(x))
        x = self .fc2(x)
        return x

#Writing the method for training
def training(net1, mydataloader, device):
    net = net1.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr = 1e-3, betas = (0.9, 0.99)
        )
    epochs = 7 #Number of epochs
    #Initialize loss graph
    loss_graph = []
    #Initialize iterations
    iteration = 0
    iterations = []
    for epoch in range(epochs):
        running_loss = 0.0
        # For loop for mydataloader to process 10000 images
        for count, batch in enumerate(mydataloader):
            # print(f"{count+1} out of {int(10000/mydataloader.batch_size)} iterations complete")
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (count+1) % 100 == 0:
                print("[epoch: %d, batch: %5d] loss: %.3f"  % (epoch+1, count+1, running_loss/100))
                loss_graph.append(running_loss/100)   #Appending loss onto a list for graphing 
                running_loss = 0.0
                iterations.append(iteration)    #Appending number of iterations passed
                iteration += 1
    return loss_graph, iterations, net

#Method for evaluating the confusion matrix
def confusionmatrix(net, mydataloader, device):
    #Set network to evaluation mode
    net = net.eval()
    correct = 0
    total = 0
    y_pred = []
    y = []
    # with torch.no_grad():
    for data in mydataloader:
        images, labels = data   #Acquiring image tensors and their respective labels from dataloader
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)   #Only interested in the labels of the predicted images
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print(predicted)
        # print(labels)
        for label,prediction in zip(labels,predicted):
            y_pred.append(prediction) # list of predicted labels
            y.append(label) # list of true labels
    # print(y)
    print('Accuracy of the network on the test images: %d %%' % (100* correct/total))
    cf_matrix = confusion_matrix(y, y_pred) # create the confusion matrix
    sns_plot = sns.heatmap(cf_matrix, annot=True, fmt='g', cbar=False) # use heatmap to demonstrate the confusion matrix
    fig = sns_plot.get_figure()
    plt.show()

#Function to create dataset given list
def dataset_appender(dataset, imgIds, coco):
    for entry in range(len(imgIds)):    #Entry is an integer index
        print(f"Round {entry+1} of {len(imgIds)}")
        img = coco.loadImgs(imgIds[entry])[0]  #img here is a dictionary
        I = img['file_name']
        dataset.append(I)
    return dataset
def datacreator():
    #Section for initializing COCO API for instance annotations
    os.chdir("hw4_RussellHo")
    # print(os.listdir())
    dataType = 'train2014'
    annFile = 'annotations/instances_{}.json'.format(dataType)
    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # # display COCO categories and supercategories
    # cats = coco.loadCats(coco.getCatIds())
    # nms=[cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))


    # get all images containing given categories
    categories_list = ['airplane', 'bus', 'cat', 'dog', 'pizza']
    dataset = []    #Initializing an empty dataset
    os.chdir('train2014/')
    for k in range(len(categories_list)):
        catIds = coco.getCatIds(catNms=categories_list[k]);
        imgIds = coco.getImgIds(catIds=catIds ); #Initializing imgIds as a list
        imgIds = imgIds[0:2000] #2000 images for each category
        dataset = dataset_appender(dataset, imgIds, coco)

    # print(len(dataset)) #Dataset should contain 10000 images given the 2014train dataset
    # #For loop for iterating over the 3 images in the given class
    # list_images = []
    # for j in range(3):
    #     img = Image.open(dataset[np.random.randint(0,len(dataset))])    #Selecting an image belonging to one of the categories at random
    #     list_images.append(img)
    # f, axarr = plt.subplots(1, 3)
    # axarr[0].imshow(list_images[0])
    # plt.axis('off')
    # axarr[1].imshow(list_images[1])
    # plt.axis('off')
    # axarr[2].imshow(list_images[2])
    # plt.axis('off')
    # plt.show()

    # #Loading a random image from the dataset
    # img = Image.open(dataset[np.random.randint(0,len(dataset))])
    # img.show()
    os.chdir("../")
    return dataset

def main():
    # #Data altering section
    # #Changing the current directory to the images directory
    # os.chdir("hw4_RussellHo/train2014")
    # #print(len(os.listdir()))
    # #Altering the image dimensions
    # for image_name in os.listdir():
    #     image = Image.open(image_name)
    #     new_image = image.resize((64, 64))
    #     new_image.save(image_name)
    # print("Image resizing section completed")

    dataset = datacreator()

    # Creating an instance from the dataloader class
    my_dataset = MyDataset(dataset)
    # Wrapping the Dataset within the DataLoader class
    mydataloader = DataLoader(my_dataset, shuffle = True, batch_size = 100, num_workers = 5)
    # # print(mydataloader.batch_size)
    os.chdir('train2014/')

    #First check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #Creating an instance from Net1
    net1 = HW4Net()
    loss_graph_1, iterations, net1 = training(net1, mydataloader, device)
    #Creating instance from Net2
    net2 = HW4Net2()
    loss_graph_2, iterations, net2 = training(net2, mydataloader, device)
    #Creating instance from Net3
    net3 = HW4Net3()
    loss_graph_3, iterations, net3 = training(net3, mydataloader, device)

    #Plotting the loss graphs
    plt.plot(iterations, loss_graph_1, label = "Net1 Loss")
    plt.plot(iterations, loss_graph_2, label = "Net2 Loss")
    plt.plot(iterations, loss_graph_3, label = "Net3 Loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Iterations")
    plt.show()

    confusionmatrix(net3, mydataloader, device)
if __name__ == "__main__":
    main()

