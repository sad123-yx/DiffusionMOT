import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
import time

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))
    return match, unmatched_O, unmatched_Q

def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)
    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )
    return ious


def iou_distance(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def v_iou_distance(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def reid_embedding_distance(tracks, feature_embs, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(feature_embs)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([emb_feature for emb_feature in feature_embs], dtype=np.float)
    track_features = np.asarray([track.smooth_emb_feature for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix

def reid_embedding_distance_features(feature_embs1,feature_embs2,metric='cosine'):
    #calculate two feature embbedings distance
    cost_matrix = np.zeros((len(feature_embs1), len(feature_embs2)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(feature_embs1, feature_embs2, metric))
    return cost_matrix

def mixed_iou_reid(iou_dist,emb_dist,iou_thresh,emb_thresh,dynamic_factor):
    mix_dist=np.zeros_like(iou_dist)
    for i in range(iou_dist.shape[0]):
        for j in range(iou_dist.shape[1]):
            iou=iou_dist[i][j]
            emb=emb_dist[i][j]
            iou_ratio = 1 / np.exp(iou / iou_thresh)
            emb_ratio = 1 / np.exp(emb / emb_thresh)
            # iou_ratio =np.log(2)/ np.log((iou / iou_thresh)+2)
            # emb_ratio = np.log(2) / np.log((emb / emb_thresh)+2)
            if iou<iou_thresh and emb<emb_thresh:
                iou_weight=iou_ratio/(iou_ratio+emb_ratio)
                emb_weight=emb_ratio/(iou_ratio+emb_ratio)
                mix_dist[i][j]=iou_weight*iou+emb_weight*emb
            elif iou<iou_thresh and emb>emb_thresh:
                mix_dist[i][j] = iou
            elif iou>iou_thresh and emb<emb_thresh:
                mix_dist[i][j]=emb
            elif iou>iou_thresh and emb>emb_thresh:
                iou_weight = iou_ratio / (iou_ratio + emb_ratio)
                emb_weight = emb_ratio / (iou_ratio + emb_ratio)
                mix_dist[i][j] = iou_weight * iou *dynamic_factor + emb_weight * emb
    return mix_dist

def linear_assignment2(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

