import argparse
import os
import time
import pickle
import pdb
import tqdm
from tqdm import tqdm

import numpy as np
import sys
from p_tqdm import p_map

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime
from image_reranking import RerankByGeometricVerification

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

datasets_names = ['roxford5k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# # network
# group = parser.add_mutually_exclusive_group(required=True)
# group.add_argument('--network-path', '-npath', metavar='NETWORK',
#                     help="pretrained network or network path (destination where network is saved)")
# group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
#                     help="off-the-shelf network, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," +
#                         " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")
#
# # test options
# parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
#                     help="comma separated list of test datasets: " +
#                         " | ".join(datasets_names) +
#                         " (default: 'oxford5k,paris6k')")
# parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
#                     help="maximum size of longer image side used for testing (default: 1024)")
# parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]',
#                     help="use multiscale vectors for testing, " +
#                     " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
# parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
#                     help="dataset used to learn whitening for testing: " +
#                         " | ".join(whitening_names) +
#                         " (default: None)")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")


def build_global_feature_dict(file_name, feature_field):
    feature = pickle.load(open(file_name, "rb"))
    features = {}
    feature_size = feature[0][feature_field].shape
    for f in feature:
        f_name = f['filename'].replace('oxford5k', 'roxford5k')
        features[f_name] = f[feature_field]
    return features, feature_size[0]

def build_local_feature_dict(file_name):
    feature = pickle.load(open(file_name, "rb"))
    feature_location = {}
    feature_descriptor = {}
    for f in tqdm(feature):
        f_name = '/data/test/roxford5k/jpg/' + f['filename'][0]
        print(f_name)
        feature_location[f_name] = f['location_np_list']
        feature_descriptor[f_name] = f['descriptor_np_list']
    return feature_location, feature_descriptor

def extract_feature(feature_dict, image_list, feature_size):
    feature_vecs = np.zeros((feature_size, len(image_list)))
    for (i, image) in enumerate(image_list):
        feature_vecs[:, i] = feature_dict[image]
    return feature_vecs

def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    # for dataset in args.datasets.split(','):
    #     if dataset not in datasets_names:
    #         raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    # download_train(get_data_root())
    # download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path


    #datasets = args.datasets.split(',')
    for dataset in datasets_names:
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join('/data', 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]

        global_feature_dict, gloabal_feature_size = build_global_feature_dict('/home/ubuntu/DELG/extract/roxf5k.delf.global',
                                                                              'global_features')
        
        # # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_feature(global_feature_dict, images, gloabal_feature_size)
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_feature(global_feature_dict, qimages, gloabal_feature_size)
        print('>> {}: Evaluating...'.format(dataset))

        # search, rank, and print
        # using cosine similarity of two vectors
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)


        compute_map_and_print(dataset, ranks, cfg['gnd'])

        feature_location, feature_descriptor = build_local_feature_dict('/home/ubuntu/DELG/extract/roxf5k.delf.local')
        new_ranks = [] #np.empty_like(ranks)
        # for i, qimage in enumerate(qimages):
        #     new_ranks[:, i] = RerankByGeometricVerification(i, ranks[:, i], scores[:, i], qimage,
        #                                       images, feature_location,
        #                                       feature_descriptor, [])

        from functools import partial
        re_rank_func = partial(RerankByGeometricVerification, input_ranks=ranks, initial_scores=scores, query_name=qimages,
                               index_names=images, feature_location=feature_location, feature_descriptor=feature_descriptor, junk_ids=[])

        new_ranks = p_map(re_rank_func, range(len(qimages)))

        new_ranks = np.concatenate(new_ranks, axis=0)

        compute_map_and_print(dataset, new_ranks, cfg['gnd'])
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
