
from PIL import Image
from io import BytesIO
from helper.feeder import Feeder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from helper import matcher

def resize_image(image, target_size=800):
    def calc_by_ratio(a, b):
        return int(a * target_size / float(b))

    size = image.size
    if size[0] < size[1]:
        w = calc_by_ratio(size[0], size[1])
        h = target_size
    else:
        w = target_size
        h = calc_by_ratio(size[1], size[0])

    image = image.resize((w, h), Image.BILINEAR)
    return image


def get_and_cache_image(image_path, basewidth=None):
    image = Image.open(image_path)
    if basewidth is not None:
        image = resize_image(image, basewidth)
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return image, imgByteArr


def get_result(feeder, query):
    pil_image = []
    byte_image = []
    for _, v in enumerate(query):
        pil, byte = get_and_cache_image(v)
        pil_image.append(pil)
        byte_image.append(byte)

    # feed and get output.
    outputs = feeder.feed_to_compare(query, pil_image)
    print('# of extracted feature (qeuery):', len(outputs[0]['descriptor_np_list']))
    print('# of extracted feature (db):', len(outputs[0]['descriptor_np_list']))

    att1 = matcher.get_attention_image_byte(outputs[0]['attention_np_list'])
    att2 = matcher.get_attention_image_byte(outputs[1]['attention_np_list'])

    side_by_side_comp_img_byte, score = matcher.get_ransac_image_byte(
        byte_image[0],
        outputs[0]['location_np_list'],
        outputs[0]['descriptor_np_list'],
        byte_image[1],
        outputs[1]['location_np_list'],
        outputs[1]['descriptor_np_list'])
    print('matching inliner num:', score)
    return side_by_side_comp_img_byte, att1, att2


if __name__ == '__main__':

    feeder_config = {
        'GPU_ID': 0,
        'IOU_THRES': 0.98,
        'ATTN_THRES': 0.17,
        'TARGET_LAYER': 'layer3',
        'TOP_K': 1000,
        #'PCA_PARAMETERS_PATH':'./output/pca/banknote_v3_hana_balanced/pca.h5',
        #'PCA_PARAMETERS_PATH':'./output/pca/ldmk/pca.h5',
        #'PCA_PARAMETERS_PATH':'./output/pca/glr2k/pca.h5',
        'PCA_DIMS': 40,
        'USE_PCA': False,
        'SCALE_LIST': [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4147, 2.0],
        #'LOAD_FROM': '../train/repo/banknote_v3_balanced_hana_alt/keypoint/ckpt/fix.pth.tar',
        #'LOAD_FROM': '../train/repo/ldmk/keypoint/ckpt/fix.pth.tar',
        'LOAD_FROM': '/home/ubuntu/DELG/train/repo/devel/finetune/ckpt/model.pth.tar',
        'ARCH': 'resnet50',
        'EXPR': 'dummy',
        'TARGET_LAYER': 'layer3',
    }

    myfeeder = Feeder(feeder_config)

    prefix = '/data/test/roxford5k/jpg/'
    query = [prefix + 'radcliffe_camera_000523.jpg', prefix + 'radcliffe_camera_000456.jpg']
    result_image_byte, att1, att2 = get_result(myfeeder, query)
    plt.figure(figsize=(16, 12))
    result_image = Image.open(BytesIO(result_image_byte))
    imshow(np.asarray(result_image), aspect='auto')
    plt.savefig('imgs/radcliffe.png')
    plt.figure(figsize=(4, 3))
    att1_image = Image.open(BytesIO(att1))
    imshow(np.asarray(att1_image), aspect='auto')
    plt.savefig('imgs/radcliffe2.png')
    plt.figure(figsize=(4, 3))
    att2_image = Image.open(BytesIO(att2))
    imshow(np.asarray(att2_image), aspect='auto')
    plt.savefig('imgs/radcliffe3.png')