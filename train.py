#########################################################################################################
#
#   ELEC 475 - Lab 2
#   Fall 2024
#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import argparse
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from torchsummary import summary
from torchvision.transforms.v2 import Resize, RandomHorizontalFlip

from model import SnoutNet
from PetNoseDataset import PetNoseDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# some default parameters, which can be overwritten by command line arguments
save_file = 'weights.pth'
n_epochs = 80
batch_size = 128
plot_file = 'plot.png'
img_dir = "oxford-iiit-pet-noses/images-original/images"
augmentations = 0

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

# Gaussian Noise Transform
class CustomGaussianNoise:
    def __init__(self, mean=0.0, std=25.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_array = np.array(img.permute(1, 2, 0))
        noise = np.random.normal(self.mean, self.std, img_array.shape).astype(np.float32)
        noise *= 0.5
        noise = noise.astype(np.int_)
        noisy_img_array = img_array + noise

        # Clip the values to stay within the [0, 255] range
        noisy_img_array = np.clip(noisy_img_array, 0, 255)

        return torch.from_numpy(noisy_img_array.astype(np.uint8)).permute(2, 0, 1)

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, validation_loss_fn=None, validation_loader=None, save_file=None, plot_file=None):
    print('training ...')

    losses_train = []
    losses_val = []
    for epoch in range(1, n_epochs+1):
        model.train()
        print('epoch ', epoch)
        loss_train = 0.0
        for data in train_loader:
            imgs = data[0]
            labels = data[1]
            if device == 'cuda':
                imgs = imgs.cuda()
                labels = labels.cuda()

            imgs = imgs.to(device=device)
            outputs = model(imgs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))

        if validation_loss_fn is not None:
            # Find the loss of the validation set
            loss_val = 0.0
            model.eval()
            for data in validation_loader:
                imgs = data[0]
                labels = data[1]

                if device == 'cuda':
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                imgs = imgs.to(device=device)
                outputs = model(imgs)
                loss = validation_loss_fn(outputs, labels)
                loss_val += loss.item()

            losses_val += [loss_val/len(validation_loader)]

            print('Epoch {}, Validation loss {}'.format(
                epoch, loss_val / len(validation_loader)))

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()

            # Limit y axis to 5000
            plt.ylim(0, 5000)

            plt.plot(losses_train, label='train')
            if validation_loss_fn is not None:
                plt.plot(losses_val, label='validation')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    global save_file, n_epochs, batch_size, plot_file, img_dir, augmentations

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', metavar='img dir', type=str, help='image directory')
    argParser.add_argument('-a', metavar='augmentations', type=int, help='which augmentations to apply')
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.d != None:
        img_dir = args.d
    if args.a != None:
        augmentations = args.a
    if args.s != None:
        save_file = args.s
    if args.e != None:
        epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p


    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('USING DEVICE ', device)

    # Define inputs (TBA)
    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    validation_loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    # Create Dataset variables with resized image and resized label coordinate
    train_data = PetNoseDataset("train_noses.txt",
                                img_dir,
                                transforms.Compose([
                                    Resize((227, 227)),
                                    transforms.Lambda(lambda img: convert_rgba_to_rgb(img))
                                ]),
                                None) # Label transform is done automatically in DataLoader
    test_data = PetNoseDataset("test_noses.txt",
                               img_dir,
                               transforms.Compose([
                                   Resize((227, 227)),
                                   transforms.Lambda(lambda img: convert_rgba_to_rgb(img))
                               ]),
                               None)
    augmented1_train_set = PetNoseDataset("train_noses.txt",
                                         img_dir,
                                         transforms.Compose([
                                             Resize((227, 227)),
                                             transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                             CustomGaussianNoise(),
                                         ]),
                                         None)
    augmented1_test_set = PetNoseDataset("test_noses.txt",
                                        img_dir,
                                        transforms.Compose([
                                            Resize((227, 227)),
                                            transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                            CustomGaussianNoise(),
                                        ]),
                                        None)
    augmented2_train_set = PetNoseDataset("train_noses.txt",
                                           img_dir,
                                           transforms.Compose([
                                               Resize((227, 227)),
                                               transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                               RandomHorizontalFlip(1),
                                           ]),
                                           "Horizontal Flip")
    augmented2_test_set = PetNoseDataset("test_noses.txt",
                                          img_dir,
                                          transforms.Compose([
                                              Resize((227, 227)),
                                              transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                              RandomHorizontalFlip(1),
                                          ]),
                                            "Horizontal Flip")
    augmented12_train_set = PetNoseDataset("train_noses.txt",
                                             img_dir,
                                             transforms.Compose([
                                                  Resize((227, 227)),
                                                  transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                                  CustomGaussianNoise(),
                                                  RandomHorizontalFlip(1),
                                             ]),
                                             "Horizontal Flip")
    augmented12_test_set = PetNoseDataset("test_noses.txt",
                                            img_dir,
                                            transforms.Compose([
                                                Resize((227, 227)),
                                                transforms.Lambda(lambda img: convert_rgba_to_rgb(img)),
                                                CustomGaussianNoise(),
                                                RandomHorizontalFlip(1),
                                            ]),
                                            "Horizontal Flip")

    # Determine which datasets to include
    if augmentations == 1:
        train_data = ConcatDataset([train_data, augmented1_train_set])
        test_data = ConcatDataset([test_data, augmented1_test_set])
    elif augmentations == 2:
        train_data = ConcatDataset([train_data, augmented2_train_set])
        test_data = ConcatDataset([test_data, augmented2_test_set])
    elif augmentations == 12:
        train_data = ConcatDataset([train_data, augmented1_train_set, augmented2_train_set, augmented12_train_set])
        test_data = ConcatDataset([test_data, augmented1_test_set, augmented2_test_set, augmented12_test_set])

    # Create the training DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print("Training on ", len(train_loader), " batches for ", n_epochs, " epochs")

    train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            scheduler=scheduler,
            device=device,
            validation_loss_fn=validation_loss_fn,
            validation_loader=test_loader,
            save_file=save_file,
            plot_file = plot_file)

###################################################################

if __name__ == '__main__':
    main()
