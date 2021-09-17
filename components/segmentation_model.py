import os
import sys
import random
import warnings
import cv2
import gc

# from tqdm import tqdm
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
import scipy as sp

# import matplotlib.pyplot as plt

from skimage.io import imread, imshow, concatenate_images
from skimage import io, transform
from skimage.measure import label, regionprops
import torch

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from PIL import Image
import math

import warnings

warnings.filterwarnings("ignore")

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_express as px

import shutil
from skimage.morphology import binary_opening, disk
import segmentation_models_pytorch as smp

# Set paths
data_root = "assets/"
path_valid = os.path.join(data_root, "validation_images")

LR = 1e-4
N_EPOCHS = 6

# Define loss function
LOSS = "BCEWithDigits"  # BCEWithDigits | FocalLossWithDigits | BCEDiceWithLogitsLoss | BCEJaccardWithLogitsLoss

# Define model
MODEL_SEG = "UNET_RESNET34ImgNet"  # UNET | IUNET | UNET_RESNET34ImgNet
FREEZE_RESNET = False  # if UNET_RESNET34ImgNet

# Decode masks in CSV
# Ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# Convert CSV masks to image for a given image name
def maskcsv_to_img(masks, img_name):
    masks_img = np.zeros((768, 768))
    masks_bin = masks.loc[masks["ImageId"] == img_name, "EncodedPixels"].tolist()
    for mask in masks_bin:
        if isinstance(mask, str):
            masks_img += rle_decode(mask)
    return np.expand_dims(masks_img, -1)


# Convert masks in a list to an image
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


# Show an image and its corresponding mask
# def imshow_mask(img, mask):
#     img = img.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = std * img + mean
#     img = np.clip(img, 0, 1)

#     mask = mask.numpy().transpose((1, 2, 0))
#     mask = np.clip(mask, 0, 1)

#     fig, axs = plt.subplots(1, 2, figsize=(10, 30))
#     axs[0].imshow(img)
#     axs[0].axis("off")
#     axs[1].imshow(mask)
#     axs[1].axis("off")


# def imshow_gt_out(img, mask_gt, mask_out):
#     img = img.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = std * img + mean
#     img = np.clip(img, 0, 1)

#     mask_gt = mask_gt.numpy().transpose((1, 2, 0))
#     mask_gt = np.clip(mask_gt, 0, 1)

#     mask_out = mask_out.numpy().transpose((1, 2, 0))
#     mask_out = np.clip(mask_out, 0, 1)

#     fig, axs = plt.subplots(1, 3, figsize=(10, 30))
#     axs[0].imshow(img)
#     axs[0].axis("off")
#     axs[0].set_title("Input image")
#     axs[1].imshow(mask_gt)
#     axs[1].axis("off")
#     axs[1].set_title("Ground truth")
#     axs[2].imshow(mask_out)
#     axs[2].axis("off")
#     axs[2].set_title("Model output")
#     plt.subplots_adjust(wspace=0, hspace=0)


# def imshow_overlay(img, mask, title=None):
#     """Imshow for Tensor."""
#     img = img.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
#     mask = mask.numpy().transpose((1, 2, 0))
#     mask = np.clip(mask, 0, 1)
#     fig = plt.figure(figsize=(6, 6))
#     plt.imshow(mask_overlay(img, mask))
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)


# def mask_overlay(image, mask, color=(0, 1, 0)):
#     """
#     Helper function to visualize mask on the top of the image
#     """
#     mask = np.dstack((mask, mask, mask)) * np.array(color)
#     weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
#     img = image.copy()
#     ind = mask[:, :, 1] > 0
#     img[ind] = weighted_sum[ind]
#     return img


# This function transforms EncodedPixels into a list of pixels
# Check our previous notebook for a detailed explanation:
# https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes
def rle_to_pixels(rle_code):
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [
        (pixel_position % 768, pixel_position // 768)
        for start, length in list(zip(rle_code[0:-1:2], rle_code[1:-2:2]))
        for pixel_position in range(start, start + length)
    ]
    return pixels


def show_pixels_distribution(df):
    """
    Prints the amount of ship and no-ship pixels in the df
    """
    # Total images in the df
    n_images = df["ImageId"].nunique()

    # Total pixels in the df
    total_pixels = n_images * 768 * 768

    # Keep only rows with RLE boxes, transform them into list of pixels, sum the lengths of those lists
    ship_pixels = df["EncodedPixels"].dropna().apply(rle_to_pixels).str.len().sum()

    ratio = ship_pixels / total_pixels
    print(f"Ship: {round(ratio, 3)} ({ship_pixels})")
    print(f"No ship: {round(1 - ratio, 3)} ({total_pixels - ship_pixels})")


class AirbusDataset(Dataset):
    def __init__(self, in_df, transform=None, mode="train"):
        grp = list(in_df.groupby("ImageId"))
        self.image_ids = [_id for _id, _ in grp]
        self.image_masks = [m["EncodedPixels"].values for _, m in grp]
        self.transform = transform
        self.mode = mode
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )  # use mean and std from ImageNet

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if (self.mode == "train") | (self.mode == "validation"):
            rgb_path = os.path.join(path_train, img_file_name)
        elif self.mode == "test":
            rgb_path = os.path.join(path_test, img_file_name)
        else:
            rgb_path = os.path.join(path_valid, img_file_name)

        img = imread(rgb_path)
        mask = masks_as_image(self.image_masks[idx])

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        if (self.mode == "train") | (self.mode == "validation"):
            return (
                self.img_transform(img),
                torch.from_numpy(np.moveaxis(mask, -1, 0)).float(),
            )
        else:
            return (
                self.img_transform(img),
                torch.from_numpy(np.moveaxis(mask, -1, 0)).float(),
            )


# Implementation from  https://github.com/ternaus/robot-surgery-segmentation
def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                mat,
                (height, width),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            if mask is not None:
                mask = cv2.warpAffine(
                    mask,
                    mat,
                    (height, width),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start : h_start + self.h, w_start : w_start + self.w, :]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[h_start : h_start + self.h, w_start : w_start + self.w, :]

        return img, mask


class CenterCrop:
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2
        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2, :]
        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[y1:y2, x1:x2, :]

        return img, mask


def compute_metrics(pred, true, batch_size=16, threshold=0.5):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)

    pred = (pred > threshold).float()
    true = (true > threshold).float()

    pred_sum = pred.sum(-1)
    true_sum = true.sum(-1)

    neg_index = torch.nonzero(true_sum == 0)
    pos_index = torch.nonzero(true_sum >= 1)

    dice_neg = (pred_sum == 0).float()
    dice_pos = 2 * ((pred * true).sum(-1)) / ((pred + true).sum(-1))

    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]

    dice = torch.cat([dice_pos, dice_neg])
    jaccard = dice / (2 - dice)

    return dice, jaccard


class metrics:
    def __init__(self, batch_size=16, threshold=0.5):
        self.threshold = threshold
        self.batchsize = batch_size
        self.dice = []
        self.jaccard = []

    def collect(self, pred, true):
        pred = torch.sigmoid(pred)
        dice, jaccard = compute_metrics(
            pred, true, batch_size=self.batchsize, threshold=self.threshold
        )
        self.dice.extend(dice)
        self.jaccard.extend(jaccard)

    def get(self):
        dice = np.nanmean(self.dice)
        jaccard = np.nanmean(self.jaccard)
        return dice, jaccard


class BCEJaccardWithLogitsLoss(nn.Module):
    def __init__(self, jaccard_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth

    def forward(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError(
                "size mismatch, {} != {}".format(outputs.size(), targets.size())
            )

        loss = self.bce(outputs, targets)

        if self.jaccard_weight:
            targets = (targets == 1.0).float()
            targets = targets.view(-1)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(-1)

            intersection = (targets * outputs).sum()
            union = outputs.sum() + targets.sum() - intersection

            loss -= self.jaccard_weight * torch.log(
                (intersection + self.smooth) / (union + self.smooth)
            )  # try with 1-dice
        return loss


class BCEDiceWithLogitsLoss(nn.Module):
    def __init__(self, dice_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError(
                "size mismatch, {} != {}".format(outputs.size(), targets.size())
            )

        loss = self.bce(outputs, targets)

        targets = (targets == 1.0).float()
        targets = targets.view(-1)
        outputs = F.sigmoid(outputs)
        outputs = outputs.view(-1)

        intersection = (outputs * targets).sum()
        dice = (
            2.0
            * (intersection + self.smooth)
            / (targets.sum() + outputs.sum() + self.smooth)
        )

        loss -= self.dice_weight * torch.log(dice)  # try with 1- dice

        return loss


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError(
                "size mismatch, {} != {}".format(outputs.size(), targets.size())
            )

        loss = self.bce(outputs, targets)

        targets = (targets == 1.0).float()
        targets = targets.view(-1)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)
        outputs = torch.where(targets == 1, outputs, 1 - outputs)

        focal = self.alpha * (1 - outputs) ** (self.gamma)
        loss *= focal.mean()

        return loss


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )

        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_loss(input, target)
        )
        return loss.mean()


# Implementation from https://github.com/timctho/unet-pytorch/
class IUNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(IUNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class IUNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(IUNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = torch.nn.Conv2d(
            prev_channel + input_channel, output_channel, 3, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class IUNet(torch.nn.Module):
    def __init__(self):
        super(IUNet, self).__init__()

        self.down_block1 = IUNet_down_block(3, 16, False)
        self.down_block2 = IUNet_down_block(16, 32, True)
        self.down_block3 = IUNet_down_block(32, 64, True)
        self.down_block4 = IUNet_down_block(64, 128, True)
        self.down_block5 = IUNet_down_block(128, 256, True)
        self.down_block6 = IUNet_down_block(256, 512, True)
        self.down_block7 = IUNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = IUNet_up_block(512, 1024, 512)
        self.up_block2 = IUNet_up_block(256, 512, 256)
        self.up_block3 = IUNet_up_block(128, 256, 128)
        self.up_block4 = IUNet_up_block(64, 128, 64)
        self.up_block5 = IUNet_up_block(32, 64, 32)
        self.up_block6 = IUNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 1, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = torch.nn.Conv2d(
            prev_channel + input_channel, output_channel, 3, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 256, True)
        self.down_block4 = UNet_down_block(256, 512, True)
        self.down_block5 = UNet_down_block(512, 1024, True)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)

        self.last_conv = torch.nn.Conv2d(64, 1, 1, padding=0)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        x = self.up_block1(self.x4, self.x5)
        x = self.up_block2(self.x3, x)
        x = self.up_block3(self.x2, x)
        x = self.up_block4(self.x1, x)
        x = self.last_conv(x)
        return x


run_id = 3

if MODEL_SEG == "IUNET":
    model = IUNet()
elif MODEL_SEG == "UNET":
    model = UNet()
elif MODEL_SEG == "UNET_RESNET34ImgNet":
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    if FREEZE_RESNET == True:
        for name, p in model.named_parameters():
            if "encoder" in name:
                p.requires_grad = False
else:
    raise NameError("model not supported")

# Model inference
model_path = "model_{fold}.pt".format(fold=run_id)
state = torch.load(str(model_path))
state = {key.replace("module.", ""): value for key, value in state["model"].items()}


model.load_state_dict(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


path_folder = os.path.join(data_root, "validation_images")
val_images_list = os.listdir(path_folder)

validation_dataset = pd.read_csv(os.path.join(data_root, "validated_images.csv"))
list_img_test = validation_dataset["ImageId"].unique()

def loading_image_and_fig(selected):

    if selected == "all":
        sample_to_load = random.sample(list(val_images_list), k=1)
    else:
        sample_to_load = random.sample(list(list_img_test), k=1)

    condition = np.where(
        validation_dataset["ImageId"].isin(sample_to_load), True, False
    )

    if condition.sum() > 0:
        validation_df = validation_dataset.loc[condition]
    else:  # Create dataframe
        validation_df = pd.DataFrame({"ImageId": sample_to_load, "EncodedPixels": None})

    if validation_df["EncodedPixels"].values[0] == None:
        text_image = "The loaded image doesn't have a ship."
    else:
        text_image = f"The loaded image has {validation_df['counts'].unique()[0]} ships on the image."

    img = io.imread(path_valid + f"/{sample_to_load[0]}")
    fig = px.imshow(img)
    fig.update_layout(
        margin=dict(t=0, b=0, r=0, l=0),
        height=300,
        width=300,
        xaxis=dict(ticks=None),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return text_image, fig, validation_df.to_dict(orient="rows")


def pytorch_segmentation(validation_df):

    loader = torch.utils.data.DataLoader(
        dataset=AirbusDataset(validation_df, transform=None, mode="valid"),
        shuffle=False,
        batch_size=8,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Display some images from loader
    images, gt = next(iter(loader))
    gt = gt.data.cpu()
    images = images.to(device)
    out = model.forward(images)
    out = ((out > 0).float()) * 255
    images = images.data.cpu()
    out = out.data.cpu()

    def imshow_gt_out_new(img, mask_gt, mask_out):

        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        mask_gt = mask_gt.numpy().transpose((1, 2, 0))
        mask_gt = np.clip(mask_gt, 0, 1)

        mask_out = mask_out.numpy().transpose((1, 2, 0))
        mask_out = np.clip(mask_out, 0, 1)

        return img, mask_gt, mask_out

    img, mask, output = imshow_gt_out_new(
        torchvision.utils.make_grid(images, nrow=1),
        torchvision.utils.make_grid(gt, nrow=1),
        torchvision.utils.make_grid(out, nrow=1),
    )

    return img, mask, output
