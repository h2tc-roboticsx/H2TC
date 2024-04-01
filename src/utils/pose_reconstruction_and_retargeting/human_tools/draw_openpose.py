import cv2
import numpy as np
import json
import os

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoColors = [[255, 0, 0], [255, 85, 0],[255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], [0, 255, 0],  
              [0, 255, 85],[0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
CocoPairsRender = CocoPairs[:-2]
# CocoPairsNetwork = [
#     (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
#     (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
#  ]  # = 19


# From https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/20d8eca4b43fe28cefc02d341476b04c6a6d6ff2/doc/output.md#pose-output-format-body_25
BODY_25_LINES = [
    [8,1,0] , # backbone
    [1,2,3,4] , # right arm
    [1, 5, 6, 7],  # left Arms
    [17, 15, 0, 16, 18], # Right eye down to left eye
    [8, 9, 10, 11, 22, 23],  #  right leg
    [11, 24],  # Right heel
    [8, 12, 13, 14, 19, 20],  # Left leg
    [14, 21],  # Left heel
]

BODY_25_Pairs = [
    (8,1), (1,0), 
    (1,2), (2, 3), (3, 4), 
    (1,5), (5, 6), (6, 7), 
    (17, 15), (15, 0), (0, 16), (16, 18),
    (8, 9), (9,10), (10, 11), (11, 22), (22, 23), 
    (11, 24), 
    (8, 12), (12, 13), (13, 14),(14, 19),(19, 20),
    (14,21),
]   # = 19

rainbow_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_RAINBOW)
Body25Colors = []
for i in range(24):
    start_index = i * 10
    end_index = (i + 1) * 10 - 1
    color = np.mean(rainbow_map[start_index:end_index], axis=0).tolist()
    print(f"Color of segment {i+1}: {color}")
    Body25Colors.append(color)

# Body25Colors = [[255, 0, 0], [255, 45, 0], 
#                 [255, 85, 0],[255, 125, 0], [255, 170, 0], 
#                 [255, 255, 0], [170, 255, 0], [125, 255, 0], 
#                 [85, 255, 0], [45, 255, 0], [0, 255, 0], [0, 255, 45],
#               [0, 255, 85], [0, 255, 125], [0, 255, 170], [0, 255, 255], [0, 170, 255], 
#               [0, 85, 255], 
#               [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], 
#               [255, 0, 85]]


def draw_openpose_coco(image_path, json_path,  img_out):
    
    npimg = cv2.imread(image_path)
    
    image_h, image_w = npimg.shape[:2]
    # npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2RGBA)
    npimg[:,:,:4] = 1.0
    centers = {}
    
    with open(json_path, "r") as f:
        humans = json.loads(f.read())
    
    humans = humans['people']
    for human_data in humans:
        # draw point
        human = np.array(human_data['pose_keypoints_2d'])
        human = human.reshape(-1,3)
        
        for i in range(18):
            # if i not in human.body_parts.keys():
            #     continue

            body_part = human[i]
            if body_part[2]<0.1:
                continue
            # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            center = (int(body_part[0] + 0.5), int(body_part[1] + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            # if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
            #     continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            h_1, h_2 = pair
            if human[h_1][2] < 0.1 or human[h_2][2]<0.1:
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    cv2.imwrite(img_out, npimg)
    return npimg


import sys
from json import load
from math import ceil

import click
import gizeh
from more_itertools.recipes import grouper, pairwise

def build_graph(lines):
    graph = {}
    for line in lines:
        for n1, n2 in pairwise(line):
            if n1 > n2:
                n1, n2 = n2, n1
            graph.setdefault(n1, set()).add(n2)
    return graph


BODY_25_GRAPH = build_graph(BODY_25_LINES)


def max_dim(doc, dim):
    return max((
        val
        for person in doc["people"]
        for numarr in person.values()
        for val in numarr[dim::3]
    ))


@click.argument("jsonin", type=click.File("r"))
@click.argument("pngout", type=click.File("wb"))
def draw_openpose_body25(imagepath, jsonin, pngout, width=1280, height=720):
    with click.open_file(jsonin, 'r') as json_file: 
        doc = load(json_file)
        if not width or not height:
            print("Warning: no width/height specified. Setting to max known + 10.", file=sys.stderr)
            width = ceil(max_dim(doc, 0)) + 10
            height = ceil(max_dim(doc, 1)) + 10
        surface = gizeh.Surface(width=width, height=height, bg_color=(1, 1, 1))
        for person in doc["people"]:
            numarr = list(grouper(person["pose_keypoints_2d"], 3))
            n = 0
            for nn in range(len(BODY_25_Pairs)):
                idx,other_idx= BODY_25_Pairs[nn]
                x1, y1, c1 = numarr[idx]
                x2, y2, c2 = numarr[other_idx]
                c = min(c1, c2)
                if c == 0:
                    continue
                line = gizeh.polyline(
                    points=[(x1, y1), (x2, y2)], stroke_width=4,
                    stroke=(Body25Colors[n][0][2]/255.0, Body25Colors[n][0][1]/255.0, Body25Colors[n][0][0]/255.0,)
                )
                line.draw(surface)
                print(idx, "--", other_idx )
                n=n+1
        with click.open_file(pngout, 'wb') as out:
            surface.write_to_png(out)

if __name__ == '__main__':
    folder = "/media/ur-5/golden_t/data/throw/004643/processed/rgbd0/"
    img_out_folder = folder + "openpose_out_img"
    if not os.path.exists(img_out_folder):
        os.makedirs(img_out_folder)
        
    image_names = []
    for i in [50,80,100]:
        name = "left_%04d" % i
        image_names.append(name)
        
    for image_name in image_names:
        image_path = folder + image_name + ".png"
        json_path = folder + "openpose_out/" + image_name + "_keypoints.json"
        img_out = img_out_folder + "/" + image_name + "_coco_openpose_render.png"
        draw_openpose_body25(image_path, json_path,  img_out)