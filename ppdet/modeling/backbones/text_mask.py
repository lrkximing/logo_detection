
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

import cv2
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.vq import vq,kmeans
import numpy as np
from paddle.vision import models
from paddleocr import PaddleOCR 
from paddle.vision.transforms import  Normalize, Compose


ocr = PaddleOCR(use_angle_cls=True, rec=False, use_gpu=True)
normalize = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
alexnet = models.alexnet()
alexnet.set_state_dict(paddle.load('./AlexNet_pretrained.pdparams'))
alexnet.eval()
clustering = 'dyhg'
iteration = 1


def cos_sim(vector_a, vector_b):
    vector_a = np.array(vector_a)[0]
    vector_b = np.array(vector_b)[0]
    return (cosine_similarity(vector_a.reshape(1,-1), vector_b.reshape(1,-1))[0][0]).round(2)


def generate_matrix(point_list):  
    length = len(point_list)
    matrix = []
    for i in range(length):
        for j in range(i + 1, length):
            matrix.append(1.0 - cos_sim(point_list[i], point_list[j]))
    return np.array(matrix)

 
def text_mask(img):
        mask_img = paddle.zeros(img.shape, dtype='float32')
        for i in range(img.shape[0]):
            image = img[i,:,:,:]
            img_tmp = paddle.squeeze(image.clone().detach(), axis=0)  
            image = np.transpose(paddle.squeeze(image, axis=0).numpy(), (1,2,0)).astype("float32") # H W C
            img_tmp = img_tmp.numpy() # C H W
            ocr_tmp = np.transpose(img_tmp, (1,2,0)).astype("float32") # H W C 

            results = ocr.ocr(ocr_tmp, rec=False)
            if results[0]!=None:
                ocr_masks = []
                ocr_mask_positions = []
                for idx in range(len(results)):
                    res = results[idx]
                    for boxes in res:
                        left = int(boxes[0][0])
                        down = int(boxes[0][1])
                        right = int(boxes[2][0])
                        up = int(boxes[2][1])
                        if left<right and down<up:
                            ocr_masks.append(cv2.resize(ocr_tmp[down:up, left:right], (224, 224)))
                            ocr_mask_positions.append(np.array([left, down, right, up]))
                            ocr_tmp[down:up, left:right] = (255, 255, 255)

                src_im = cv2.resize(ocr_tmp, (224, 224))
                
                if len(ocr_masks)>0:
                    if clustering=='km': 
                        feature_background = np.array((alexnet(paddle.unsqueeze(paddle.to_tensor(np.transpose(src_im, (2,0,1)).astype("float32")), axis=0))))[0].tolist()
                        feature = [feature_background]
                        for mask in ocr_masks:
                            feature.append(np.array((alexnet(paddle.unsqueeze(paddle.to_tensor(np.transpose(mask, (2,0,1)).astype("float32")), axis=0))))[0].tolist())
                        centerpoints,_ = kmeans(np.array(feature), 2) #分成两类（logo，text）
                        clxs, dist = vq(np.array(feature), centerpoints)
                        for idx,clx in enumerate(clxs):
                            if clx !=clxs[0]:
                                image[ocr_mask_positions[idx-1][1]:ocr_mask_positions[idx-1][3], ocr_mask_positions[idx-1][0]:ocr_mask_positions[idx-1][2]] = (255, 255, 255)
                   
                    elif clustering == 'dyhg': 
                        feature_background = np.array(alexnet(paddle.unsqueeze(paddle.to_tensor(np.transpose(src_im, (2,0,1)).astype("float32")), axis=0)))[0].tolist()
                        feature = [feature_background]
                        for mask in ocr_masks:
                            feature.append(np.array(alexnet(paddle.unsqueeze(paddle.to_tensor(np.transpose(mask, (2,0,1)).astype("float32")), axis=0)))[0].tolist())

                        for itea in range(iteration):
                            centerpoints,_ = kmeans(np.array(feature), 2)
                            clxs, dist = vq(np.array(feature), centerpoints)
                            hyperedges = list(set(clxs))
                            hyperedge_features = []

                            for e in range(len(hyperedges)):
                                tmp_list = []
                                for idx,clx in enumerate(clxs):
                                    if clx == hyperedges[e]:
                                        tmp_list.append(idx)
                                    if idx == len(clxs) - 1:
                                        feature_tmp = np.zeros((1,1000)) 
                                        for j in range(len(tmp_list)):
                                            feature_tmp = feature_tmp + np.array(feature[tmp_list[j]]).reshape(1,-1)
                                        hyperedge_features.append(feature_tmp/len(tmp_list))

                            for idx0,clx0 in enumerate(clxs):
                                dis = 1 - cosine_similarity(np.array(feature[idx0]).reshape(1,-1), hyperedge_features[clx0-1]).round(2)
                                if dis > 0.05:
                                    clxs[idx0] = 2

                            feature_copy = feature.copy()
                            for idx1,clx1 in enumerate(clxs):
                                if clx1 != 2:
                                    for j in range(len(clxs)):
                                        tmp_vertex = np.zeros((1,1000))
                                        if clxs[j] == clx1:
                                            w = ((1 + cosine_similarity(np.array(feature_copy[idx1]).reshape(1,-1), np.array(feature_copy[j]).reshape(1,-1)).round(2))/2)[0][0]
                                            tmp_vertex = tmp_vertex + np.array(feature_copy[j]).reshape(1,-1) * w
                                        if j == len(clxs) - 1:
                                            feature[idx1] = tmp_vertex.tolist()[0]
                                        
                        for idx2,clx2 in enumerate(clxs):
                            if clx2 !=clxs[0] or (idx2 > 0 and clx2 == 2):
                                image[ocr_mask_positions[idx2-1][1]:ocr_mask_positions[idx2-1][3], ocr_mask_positions[idx2-1][0]:ocr_mask_positions[idx2-1][2]] = (255, 255, 255)
                        
            image = paddle.to_tensor(np.transpose(image, (2,0,1)).astype("float32"))
            image = normalize(image/255)
            mask_img[i,:,:,:] = image
        return mask_img