import os, sys, shutil, argparse, subprocess, time, json, glob
from multiprocessing import Pool

import os.path as osp
import torch, torchvision
from PIL import Image
from torchvision import transforms
import numpy as np

import cv2

def run_deeplab_v3(img_dir, img_shape, out_dir, batch_size=16, img_extn='jpg'):
    '''
    Runs DeepLabv3 to get a person segmentation mask on each img in img_dir.
    
    - img_shape : (H x W)
    '''
    print('Running DeepLabv3 to compute person mask...')
    H, W = img_shape
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).to(device)
    model.eval()
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    img_path = img_dir
    all_img_paths = sorted(glob.glob(os.path.join(img_path + '/*.'  + img_extn)))
    img_names = ['.'.join(f.split('/')[-1].split('.')[:-1]) for f in all_img_paths]
    out_path = out_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_mask_paths = [os.path.join(out_path, f + '.png') for f in img_names]
    # print(all_mask_paths)

    num_imgs = len(img_names)
    num_batches = (num_imgs / batch_size) + 1
    sidx = 0
    eidx = min(num_imgs, batch_size)
    cnt = 1
    while sidx < num_imgs:
        # print(sidx)
        # print(eidx)
        # batch
        # print('Batch %d / %d' % (cnt, num_batches))
        img_path_batch = all_img_paths[sidx:eidx]
        mask_path_batch = all_mask_paths[sidx:eidx]
        B = len(img_path_batch)
        img_batch = torch.zeros((B, 3, H, W))
        for bidx, cur_img_path in enumerate(img_path_batch):
            input_image = Image.open(cur_img_path)
            input_tensor = preprocess(input_image)
            img_batch[bidx] = input_tensor
        img_batch = img_batch.to(device)
        # print(img_batch.size())

        # eval and save
        with torch.no_grad():
            output = model(img_batch)['out']
        seg = torch.logical_not(output.argmax(1) == 15).to(torch.float) # the max probability is the person class
        seg = seg.cpu().numpy()
        for bidx in range(B):
            person_mask = (seg[bidx]*255.0).astype(np.uint8)
            out_img = Image.fromarray(person_mask)
            out_img.save(mask_path_batch[bidx])


        # # create a color pallette, selecting a color for each class
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")
        # # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(seg[0].byte().cpu().numpy()).resize(input_image.size)
        # r.putpalette(colors)
        # import matplotlib.pyplot as plt
        # plt.imshow(r)
        # plt.show()

        sidx = eidx
        eidx = min(num_imgs, sidx + batch_size)
        cnt += 1