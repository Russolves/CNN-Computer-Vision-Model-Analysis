# Import the necessary modules
import torch
from PIL import Image
from torchvision import transforms as tvt

# Creating the class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        labels = []
        for i in range(5):
            labels += [i for j in range(2000)]  #Append labels based on how data was loaded
        self.labels = labels
    def __len__(self):
        # Return the total number of images
        return len(self.dataset)
    def __getitem__(self, index):
        img = Image.open(self.dataset[index])
        img_tensor = tvt.ToTensor()(img) #Convert Image to tensors (C x H x W)
        #Ensure all images possess same channels (changing 1 channel images to 3)
        if img_tensor.size()[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        # Apply transformations to the image
        transform1 = tvt.RandomAffine(degrees = 30, translate = (0.2, 0.2))
        transform2 = tvt.ColorJitter(brightness = (0.7, 1), saturation = (0.5, 1), contrast = (0.4, 1))
        transform3 = tvt.RandomHorizontalFlip()
        transform = tvt.Compose([transform1, transform2, transform3])
        # Transform the non-oblique image
        trans_tensor = transform(img_tensor)
        int_label = self.labels[index]
        # Return the tuple: (augmented tensor, integer label)
        return trans_tensor, int_label
