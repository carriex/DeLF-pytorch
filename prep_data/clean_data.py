'''clean google landmark dataset v2'''

# https://arxiv.org/pdf/1906.11874.pdf

from argparse import ArgumentParser
import os
import networkx as nx
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def gem(x, p=3, eps=1e-6):
    '''generalized mean pooling'''
    m =  nn.AvgPool2d((x.size(-2), x.size(-1)))
    m.to(device)
    return m(x.clamp(min=eps).pow(p)).pow(1/p)

def set_seed(seed=1234567):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GoogleLandmark(Dataset):
    '''dataset without low frequency images'''
    def __init__(self, data_dir, data_csv, label_column):
        self.data_dir = data_dir
        self.data_csv = pd.read_csv(data_csv)
        self.label_column = label_column
        self.labels = self.filter_infrequent_class()
        self.samples = self.get_samples()
        self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

    def filter_infrequent_class(self, min_count=4):
        class_count = self.data_csv.groupby(self.label_column).count().id.to_numpy()
        freq_classes = np.arange(len(class_count))[class_count >= min_count]
        return freq_classes

    def get_samples(self):
        sample_ids = self.data_csv.set_index(self.label_column).sort_values(by=self.label_column)
        freq_sample_ids = sample_ids.loc[self.labels].id.to_numpy()
        freq_sample_labels = sample_ids.loc[self.labels].index.to_numpy()
        return np.array((freq_sample_ids, freq_sample_labels)).T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''return (img_id, transformed feature, label)'''
        img_id, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, str(label), str(img_id)+'.jpg')
        img = Image.open(img_path)
        return img_id, self.img_transform(img), label


def create_img_pairs(cosine_similarity, threshold=0.5, num_pairs=100):
    '''return list of image ids in the connected components'''
    idx = (cosine_similarity > threshold).reshape(cosine_similarity.shape)
    x = np.arange(cosine_similarity.shape[0])
    y = np.arange(cosine_similarity.shape[1])
    x_idx, y_idx = np.meshgrid(x, y)
    # pair of photos above threshold
    pairs = np.array((x_idx, y_idx)).T[idx]

    # build the graph given cosine similarity
    G = nx.Graph()
    for pair in pairs:
        x, y = pair
        G.add_edge(x, y)

    max_connected_component = max(nx.connected_components(G), key=len)

    # return at most 100 pairs from the max connected component
    ids = set()
    for node in max_connected_component:
        for neighbor in G.adj[node]:
            ids.add(node)
            ids.add(neighbor)
            if len(list(ids)) == num_pairs * 2:
                return np.array(list(ids))
    return np.array(list(ids))


def create_clean_data(label, images_per_label):
    '''return (img_id, landmark_id) pairs given a landmark_id'''

    # build matrix for cosine similarity
    features = np.array(images_per_label[label]['feature'])
    img_ids = images_per_label[label]['id']
    # class with no image
    if len(img_ids) == 0:
        return None
    features_norm = features / np.linalg.norm(features, axis=1)[:, None]
    cosine_similarity = np.matmul(features_norm, features_norm.T)
    img_idx = create_img_pairs(cosine_similarity)
    # class with no connected components
    if len(img_idx) == 0:
        return None
    img_ids = np.array(img_ids)[img_idx]
    labels = np.ones_like(img_idx) * int(label)
    return np.array((img_ids, labels)).T

if __name__ == "__main__":

    # set seed
    set_seed()
    # add arguments
    argparse = ArgumentParser("Clean Google Landmark Dataset v2 using ResNet 50 features")

    argparse.add_argument("-c", "--csv_file", help="csv file containing instance to label mapping",
                          dest="csv_file", default="/data/google-landmark/csv/train.csv")
    argparse.add_argument("-o", "--output_file", help="output csv file name for the clean training data", dest="output_file",
                          default='/data/google-landmark/csv/train-clean.csv')
    argparse.add_argument("-l", "--label_column", help="column name storing label id",
                          dest="label_column",
                          default='landmark_id')
    argparse.add_argument("-f", "--feature_file", help="output npy file name for the extracted ResNet features of the image",
                          dest="feature_file",
                          default='/data/google-landmark/train-feature.npy')
    argparse.add_argument("-d", "--data_dir", help="root image folder", dest="data_dir",
                          default="/data/google-landmark/org/train")
    argparse.add_argument("-e", "--extension", help="file extension. default: jpg", dest="extension",
                          default=".jpg")

    args = argparse.parse_args()
    dataset = GoogleLandmark(data_dir=args.data_dir,
                             data_csv=args.csv_file,
                             label_column=args.label_column)

    # define pretrained model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    resnet50 = models.resnet50(pretrained=True)
    # extract feature from last conv layer
    modules = list(resnet50.children())[:-2]
    resnet50_conv = nn.Sequential(*modules)
    for p in resnet50_conv.parameters():
        p.requires_grad = False
    resnet50_conv.to(device)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=10)

    # extract image feature
    img_features = []
    for i, data in enumerate(tqdm(dataloader)):
        ids, inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float)
        output_batch = resnet50_conv(inputs)
        output_batch = gem(output_batch)
        img_features.append(output_batch.squeeze().cpu())

    # num_samples x 1024
    img_features = torch.cat(img_features, dim=0).numpy()

    np.save(args.feature_file, img_features)

    print("Feature extracted")

    img_features = np.load(args.feature_file)

    images_per_label = {}
    for label in dataset.labels:
        images_per_label[label] = {'id': [],
                                   'feature': []}

    for i, feature in enumerate(img_features):
        id, label = dataset.samples[i]
        images_per_label[label]['id'].append(id)
        images_per_label[label]['feature'].append(feature)

    img_idx = []
    for label in tqdm(dataset.labels):
        img_idx.append(create_clean_data(label, images_per_label))

    # img_idx, labels
    images = np.concatenate([idx for idx in img_idx if idx is not None ])
    clean_data = {'id': images[:, 0],
                  args.label_column: images[:, 1].astype(float).astype(int)}

    # save all (image_id, landmark_id) pair to a csv file
    clean_data_df = pd.DataFrame.from_dict(clean_data).set_index('id')
    clean_data_df.to_csv('/data/google-landmark/csv/train-clean.csv')
