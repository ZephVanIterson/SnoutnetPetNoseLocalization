
import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from model import SnoutNet
from PetNoseDataset import PetNoseDataset
from torchvision.transforms.v2 import Resize
from torch.utils.data import DataLoader, ConcatDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

save_file = 'weights.pth'
save_file_no_aug = 'Pretrained/weights_no_aug.pth'
save_file_hflip = 'Pretrained/weights_hflip.pth'
save_file_noise = 'Pretrained/weights_noise.pth'
save_file_hflip_noise = 'Pretrained/weights_hflip_noise.pth'
img_dir = "oxford-iiit-pet-noses/images-original/images"
batch_size = 1
augmentations = 0

# Custom Gaussian noise transform
class CustomGaussianNoise:
    def __init__(self, mean=0.0, std=25.0):  # Adjust std for desired noise level
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_array = np.array(img.permute(1, 2, 0))  # Convert PIL Image to a NumPy array
        noise = np.random.normal(self.mean, self.std, img_array.shape).astype(np.float32)
        noise *= 0.5
        noise = noise.astype(np.int_)
        noisy_img_array = img_array + noise

        # Clip the values to stay within the [0, 255] range
        noisy_img_array = np.clip(noisy_img_array, 0, 255)

        return torch.from_numpy(noisy_img_array.astype(np.uint8)).permute(2, 0, 1)

# Define a transform to handle images with varying channels
def convert_rgba_to_rgb(image):
    # If the image has 4 channels (RGBA), take only the first 3 (RGB)
    if image.shape[0] >= 4:
        image = image[:3, :, :]  # Keep only RGB, discard the Alpha channel
    elif image.shape[0] == 1:
        # Copy to all 3 channels
        image = torch.cat((image, image, image), 0)
    elif image.shape[0] == 2:
        image = torch.cat((image[0], image[1], image[0]), 0)
    return image


def main():
    global save_file, batch_size, img_dir, augmentations

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', metavar='img dir', type=str, help='image directory')
    argParser.add_argument('-a', metavar='augmentations', type=int, help='which augmentations to apply')
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.d != None:
        img_dir = args.d
    if args.a != None:
        augmentations = args.a
    if args.s != None:
        save_file = args.s
    if args.b != None:
        batch_size = args.b

    print('running testing ...')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    # load datasets
    test_data = PetNoseDataset("test_noses.txt",
                               img_dir,
                               transforms.Compose([
                                   Resize((227, 227)),
                                   transforms.Lambda(lambda img: convert_rgba_to_rgb(img))
                               ]),
                               None)
    
    test_data_hflip = PetNoseDataset("test_noses.txt",
                               img_dir,
                               transforms.Compose([
                                   Resize((227, 227)),
                                   transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                   transforms.RandomHorizontalFlip(1)
                               ]),
                               "Horizontal Flip")
    
    test_data_noise = PetNoseDataset("test_noses.txt",
                                 img_dir,
                                 transforms.Compose([
                                      Resize((227, 227)),
                                      transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                      CustomGaussianNoise()
                                 ]),
                                 None)
    
    test_data_hflip_noise = PetNoseDataset("test_noses.txt",
                                 img_dir,
                                 transforms.Compose([
                                      Resize((227, 227)),
                                      transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                      CustomGaussianNoise(),
                                      transforms.RandomHorizontalFlip(1)
                                 ]),
                                 "Horizontal Flip")

    if augmentations == 0:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        save_file = save_file_no_aug
    elif augmentations == 1:
        test_data = ConcatDataset([test_data, test_data_noise])
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        save_file = save_file_noise
    elif augmentations == 2:
        test_data = ConcatDataset([test_data, test_data_hflip])
        test_loader = DataLoader( test_data, batch_size=batch_size, shuffle=False)
        save_file = save_file_hflip
    elif augmentations == 12:
        test_data = ConcatDataset([test_data, test_data_hflip, test_data_noise, test_data_hflip_noise])
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        save_file = save_file_hflip_noise
    else:
        print("Invalid choice")
        return
    
    # Create the model
    model = SnoutNet()
    model.to(device)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(save_file))
    else:
        model.load_state_dict(torch.load(save_file, map_location=torch.device('cpu')))

    model.eval()

    #Calculate the localization accuracy statistics, i.e. the minimum, mean, maximum, and standard deviation of the Euclidean distance from your estimated pet nose locations, to the ground truth pet nose locations.
    min_distance = 1000
    min_index = -1
    max_distance = 0
    max_index = -1
    total_distance = 0
    mean_distance = 0
    std_distance = 0
    count = 0
    distance_list = []
    total_loss = 0
    average_loss = 0

    for data in test_loader:
        image, label = data
        image = image.to(device)
        label = label.to(device)
        output = model(image)

        # Calculate the Euclidean distance between the predicted and ground truth coordinates
        distance = float(torch.dist(output, label, p=2))

        if distance < min_distance:
            min_distance = distance
            min_index = count
        if distance > max_distance:
            max_distance = distance
            max_index = count

        loss = torch.nn.MSELoss()(output, label)
        total_loss += loss.item()

        total_distance += distance
        count += 1
        distance_list.append(distance)
    
    mean_distance = total_distance/count
    std_distance = np.std(distance_list)
    average_loss = total_loss/count

    print("Min distance: ", min_distance, " at index: ", min_index)
    print("Max distance: ", max_distance, " at index: ", max_index)
    print("Mean distance: ", mean_distance)
    print("Standard deviation: ", std_distance)
    print("Average loss: ", average_loss)

    #test chosen image
    idx = 0
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)

        #get image from data loader for index
        for i, data in enumerate(test_loader):
            if i == idx:
                image, label = data
                break

        image = image.to(device)
        label = label.to(device)
        output = model(image)

        #get distance between output and label
        distance = float(torch.dist(output, label, p=2))
        print("Distance between output and label: ", distance)
        
        # Ensure the tensor is in the right shape (H, W, C)
        image_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)

        # If the values are outside the range [0, 1], scale them
        if image_np.max() > 1:
            image_np = image_np / 255.0

        #Add both dots(output and label) to image
        output = output[0].cpu().detach().numpy()
        label = label[0].cpu().detach().numpy()

        # Scale the coordinates back to the original image size
        output[0] = output[0] * 227 / 227
        output[1] = output[1] * 227 / 227

        label[0] = label[0] * 227 / 227
        label[1] = label[1] * 227 / 227

        # Plot the image
        plt.scatter(label[0], label[1], color='red')
        plt.scatter(output[0], output[1], color='blue')


        plt.imshow(image_np)
        plt.show()



###################################################################

if __name__ == '__main__':
    main()



