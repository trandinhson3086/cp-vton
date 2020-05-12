import argparse
import numpy as np
import os
import json
from torchvision import transforms
from torch.utils import data
from tqdm import tqdm
from annoy import AnnoyIndex
from PIL import Image
import os
import torchvision.transforms.functional as TF
import torch
from torchvision.utils import save_image
from torch import distributed
import sys
import torchvision

#file descriptor loading bug
torch.multiprocessing.set_sharing_strategy('file_system')

def handler(signal_received, frame):
    # Handle any cleanup here
    print(signal_received)
    distributed.destroy_process_group()
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    sys.exit(0)

parser = argparse.ArgumentParser()

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    SAVE_DIR = "result/product_embeddings/"
    with open(os.path.join('data/identity_embedding.json')) as outfile:
        data = json.load(outfile)
        product_ids = data["im_names"]
        o_embeddings = data["embeddings_from_model"]
        p_embeddings = data["embeddings_from_product"]

    def build_tree(vectors, dim=64):
        a = AnnoyIndex(dim, 'euclidean')
        for i, v in enumerate(vectors):
            a.add_item(i, v)
        a.build(-1)
        return a

    product_tree = build_tree(p_embeddings)
    outfit_tree = build_tree(o_embeddings)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)


    for i in tqdm(range(100)):
        closest_items = outfit_tree.get_nns_by_vector(p_embeddings[i], 99)
        images = []

        image = Image.open(os.path.join("data/train/image",  product_ids[i])).convert('RGB')
        image = transforms.Resize(size=(256, 256))(image)
        image = TF.to_tensor(image).unsqueeze(0)
        images.append(image)

        for index in closest_items:
            id = product_ids[index]

            image = Image.open(os.path.join("data/train/image", id)).convert('RGB')
            image = transforms.Resize(size=(256, 256))(image)
            image = TF.to_tensor(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images, dim=0)
        images = torchvision.utils.make_grid(images, nrow=10)
        save_image(images, SAVE_DIR + "test_" + product_ids[i] + "_outfit_prod.jpg")


    for i in tqdm(range(100)):
        closest_items = outfit_tree.get_nns_by_vector(o_embeddings[i], 99)
        images = []

        image = Image.open(os.path.join("data/train/image", product_ids[i])).convert('RGB')
        image = transforms.Resize(size=(256, 256))(image)
        image = TF.to_tensor(image).unsqueeze(0)
        images.append(image)

        for index in closest_items:
            id = product_ids[index]

            image = Image.open(os.path.join("data/train/image", id)).convert('RGB')
            image = transforms.Resize(size=(256, 256))(image)
            image = TF.to_tensor(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images, dim=0)
        images = torchvision.utils.make_grid(images, nrow=10)
        save_image(images, SAVE_DIR + "test_" + product_ids[i] + "_outfit_outfit.jpg")

