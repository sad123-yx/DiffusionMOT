import numpy as np
from collections import deque
import time
import torch
import torch.nn.functional as F 
import torchvision
from copy import deepcopy
from yolox.tracker import matching
from detectron2.structures import Boxes
from yolox.utils.box_ops import box_xyxy_to_cxcywh
from yolox.utils.boxes import xyxy2cxcywh
from torchvision.ops import box_iou,nms
from yolox.utils.cluster_nms import cluster_nms

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

from fast_reid.fast_reid_interfece import FastReIDInterface
from yolox.tracker.gmc import GMC
from yolox.utils.visualize import plot_tracking_xyxy
import matplotlib.pyplot as plt
import cv2
import os
from yolox.tracker.sportsmot_det import sportsmot_add

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score,feature=None,feat_history=50):

        # wait activate
        self.buffer_size=30
        self.xywh_omemory = deque([], maxlen=self.buffer_size)
        self.xywh_pmemory = deque([], maxlen=self.buffer_size)
        self.xywh_amemory = deque([], maxlen=self.buffer_size)

        self.conds = deque([], maxlen=5)
        self.conds_scaled = deque([], maxlen=5)
        self._tlwh = np.asarray(tlwh)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.xywh=self.tlwh_to_xywh()
        self.score = score
        self.tracklet_len = 0
        self.scale=0.7407407407407407

        self.smooth_emb_feature = feature

    def update_feature(self, feat, alpha=0.95):
        self.curr_feat = feat
        self.smooth_emb_feature = alpha * self.smooth_emb_feature + (1 - alpha) * feat
        self.smooth_emb_feature /= np.linalg.norm(self.smooth_emb_feature)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)): #更新均值和协方差
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_diff(stracks, model, img_h, img_w,scale):
        if len(stracks) > 0:
            dets = np.asarray([st.xywh.copy() for st in stracks]).reshape(-1, 4)
            dets = dets/scale

            dets[:, 0::2] = dets[:, 0::2] / img_w
            dets[:, 1::2] = dets[:, 1::2] / img_h

            conds = [st.conds.copy() for st in stracks]
            len_conds = len(conds)
            for i in range(len_conds):
                temp = conds[i]
                len_deque = len(temp)
                for j in range(len_deque):
                    conds[i][j] = conds[i][j] / scale

            multi_track_pred = model.generate(conds, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            track_pred = multi_track_pred.mean(0)
            track_pred = track_pred + dets
            track_pred[:, 0::2] = track_pred[:, 0::2] * img_w
            track_pred[:, 1::2] = track_pred[:, 1::2] * img_h
            track_pred[:, 0] = track_pred[:, 0] - track_pred[:, 2] / 2
            track_pred[:, 1] = track_pred[:, 1] - track_pred[:, 3] / 2
            track_pred=track_pred*scale

            for i, st in enumerate(stracks):
                st._tlwh = track_pred[i]

                ret = np.asarray(st._tlwh).copy()
                ret[:2] += ret[2:] / 2
                st.xywh=ret

                ret = st._tlwh.copy()
                ret[2:] += ret[:2]

                st.xywh_pmemory.append(st.xywh.copy())
                st.xywh_amemory.append(st.xywh.copy())

                tmp_delta_bbox = st.xywh.copy() - st.xywh_amemory[-2].copy()
                tmp_conds = np.concatenate((st.xywh.copy(), tmp_delta_bbox))
                st.conds.append(tmp_conds)

                scaled_tmp_conds=tmp_conds/scale
                st.conds_scaled.append(scaled_tmp_conds)

    def pass_id(self,n):
        for i in range(n):
            self.track_id = self.next_id()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        if  kalman_filter:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        if kalman_filter is None:
            self.xywh_omemory.append(self.xywh.copy())
            self.xywh_pmemory.append(self.xywh.copy())
            self.xywh_amemory.append(self.xywh.copy())

            delta_bbox = self.xywh.copy() - self.xywh.copy()
            tmp_conds = np.concatenate((self.xywh.copy(), delta_bbox))
            self.conds.append(tmp_conds)

            scaled_tmp_conds = tmp_conds / self.scale
            self.conds_scaled.append(scaled_tmp_conds)


    def re_activate(self, new_track,kalman_filter, frame_id, new_id=False,alpha=None):
        if kalman_filter:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
            )

        self._tlwh = new_track.tlwh

        #update xywh
        ret = np.asarray(self._tlwh).copy()
        ret[:2] += ret[2:] / 2
        self.xywh = ret

        if kalman_filter is None:
            self.xywh_omemory.append(self.xywh.copy())
            self.xywh_amemory[-1] = self.xywh.copy()

            tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
            tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds

            scaled_tmp_conds = tmp_conds / self.scale
            self.conds_scaled[-1] = scaled_tmp_conds

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track,kalman_filter, frame_id,update_feature=False):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh

        #update xywh
        ret = np.asarray(self._tlwh).copy()
        ret[:2] += ret[2:] / 2
        self.xywh = ret

        if kalman_filter:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        if kalman_filter is None:
            self.xywh_omemory.append(self.xywh.copy())
            self.xywh_amemory[-1] = self.xywh.copy()

            if self.is_activated == True:
                tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
                tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
                self.conds[-1] = tmp_conds
                scaled_tmp_conds = tmp_conds / self.scale
                self.conds_scaled[-1] = scaled_tmp_conds

            else:
                tmp_delta_bbox = self.xywh.copy() - self.xywh_omemory[-2].copy()
                tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
                self.conds[-1] = tmp_conds
                scaled_tmp_conds = tmp_conds / self.scale
                self.conds_scaled[-1] = scaled_tmp_conds

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_feature(new_track.smooth_emb_feature)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def tlwh_to_xywh(self):
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



class DiffusionMOTTracker(object):
    def __init__(self, model,tensor_type,args,video_name=None,conf_thresh=0.7, det_thresh=0.6, nms_thresh_3d=0.7, nms_thresh_2d=0.75,
                 interval=5, detections=None,scale=None):
        self.frame_id = 0
        self.backbone=model.backbone
        self.feature_projs=model.projs
        self.diffusion_model=model.head
        self.feature_extractor=self.diffusion_model.head.box_pooler
        self.det_thresh = det_thresh
        self.association_thresh = conf_thresh

        self.nms_thresh_2d=nms_thresh_2d
        self.nms_thresh_3d=nms_thresh_3d

        self.pre_features=None
        self.data_type=tensor_type
        self.detections=detections

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.max_time_lost = 30
        self.kalman_filter = KalmanFilter()
        self.use_det = False
        self.args = args
        self.same_thresh = 0.8

        self.repeat_times=1
        self.dynamic_time=True

        self.high_det_thresh = 0.6
        self.low_det_thresh  = 0.4

        self.use_KalmanFilter=False
        self.sampling_steps = 1
        self.num_boxes = 2000

        #parallel sample
        self.use_parallel= False   #args.use_parallel_sample
        self.parallel_sweep = 1    #[1,2,4] sliding windows
        self.inference_time_range=1
        self.step_num=20

        #mixed_iou
        #self.use_mixed_dist=True
        self.dynamic_factor=0.6

        self.alpha_fixed_emb=0.95
        self.use_dynamic_appearance=True
        self.dynamic_appearance_type="det"
        self.track_t=400

        self.reid = self.args.reid
        self.video_name = video_name
        self.val = self.args.track_val
        self.scale = scale
        self.GMC = False

        # ReID
        self.proximity_thresh = self.args.proximity_thresh
        self.appearance_thresh = self.args.appearance_thresh
        if self.reid:
            self.encoder = FastReIDInterface(self.args.fast_reid_config, self.args.fast_reid_weights, self.args.devices)
        if video_name and self.GMC:
            self.gmc = GMC(method=self.args.cmc_method, verbose=[video_name, self.val])

    def update(self,predict_track_model,cur_image,raw_img):
        self.frame_id += 1

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        self.predict_track_model=predict_track_model
        cur_features,mate_info=self.extract_feature(cur_image=cur_image)
        mate_shape,mate_device,mate_dtype=mate_info
        self.diffusion_model.device=mate_device
        self.diffusion_model.dtype=mate_dtype
        b,_,h,w=mate_shape
        images_whwh=torch.tensor([w, h, w, h], dtype=mate_dtype, device=mate_device)[None,:].expand(4*b,4)

        if self.frame_id==1:
            if self.pre_features is None:
                self.pre_features=cur_features
            inps=self.prepare_input(self.pre_features,cur_features) #pre_f=(pre,cur) cur_f=(cur,cur)!!!

            diffusion_outputs,conf_scores,association_time=self.diffusion_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                dynamic_time=self.dynamic_time,track_candidate=self.repeat_times)

            _,_,detections=self.diffusion_postprocess(diffusion_outputs,conf_scores,conf_thre=self.association_thresh,nms_thre=self.nms_thresh_3d)

            detections=self.diffusion_det_filt(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)
            feature_emb=self.get_feature_emb(raw_img,detections[:, 0:4])
            if self.use_dynamic_appearance:
                dets_alpha=self.compute_dynamic_appearance(detections[:, 5],self.dynamic_appearance_type)
            for i, det in enumerate(detections):
                track = STrack(STrack.tlbr_to_tlwh(det[:4]), det[5], feature_emb[i])  # top left_x,y,w,h
                if self.use_KalmanFilter:
                    track.activate(self.kalman_filter, self.frame_id)
                else:
                    track.activate(None,self.frame_id)
                self.tracked_stracks.append(track)
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]
            return output_stracks,association_time
        else:
            ref_bboxes=[STrack.tlwh_to_tlbr(track._tlwh) for track in self.tracked_stracks]
            inps=self.prepare_input(self.pre_features,cur_features)
            if len(ref_bboxes)>0:
                bboxes=box_xyxy_to_cxcywh(torch.tensor(np.array(ref_bboxes))).type(self.data_type).reshape(1,-1,4).repeat(2,1,1)
            else:
                bboxes=None

            diffusion_outputs,conf_scores,association_time=self.diffusion_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                ref_targets=bboxes,dynamic_time=self.dynamic_time,track_candidate=self.repeat_times,diffusion_t=self.track_t)
            diffusion_ref_detections,diffusion_track_detections,detections=self.diffusion_postprocess(diffusion_outputs,
                                                                                                      conf_scores,
                                                                                                      conf_thre=self.association_thresh,
                                                                                                      nms_thre=self.nms_thresh_3d)
            detections=self.diffusion_det_filt(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)
            diffusion_ref_detections,diffusion_track_detections=self.diffusion_track_filt(diffusion_ref_detections,
                                                                                          diffusion_track_detections,
                                                                                          conf_thre=self.det_thresh,
                                                                                          nms_thre=self.nms_thresh_2d)

            ####### no sc start
            # nms_out_index = torchvision.ops.batched_nms(
            #     diffusion_ref_detections[:, :4],
            #     diffusion_ref_detections[:, 5],
            #     diffusion_ref_detections[:, 6],
            #     self.nms_thresh_2d,
            # )
            #diffusion_ref_detections=diffusion_ref_detections[nms_out_index].cpu().numpy()
            #diffusion_track_detections=diffusion_track_detections[nms_out_index].cpu().numpy()
            ###### no sc end

            #sc start
            feature_emb = self.get_feature_emb(raw_img, diffusion_ref_detections[:, :4])
            dists = matching.iou_distance(ref_bboxes, diffusion_ref_detections[:, :4])  # dist：1-Iou_matrix
            ious_dists_mask = (dists > self.proximity_thresh)
            emb_dists = matching.reid_embedding_distance(self.tracked_stracks, feature_emb) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)
            # fix
            if len(matches) > 0:
                # fix position with detection result
                dists_fix = matching.iou_distance(diffusion_track_detections[matches[:, 1], :4], detections[:, :4])
                feature_fix_emb1 = self.get_feature_emb(raw_img, diffusion_track_detections[matches[:, 1], :4])
                feature_fix_emb2 = self.get_feature_emb(raw_img, detections[:, :4])
                emb_dists_fix = matching.reid_embedding_distance_features(feature_fix_emb1, feature_fix_emb2) / 2.0
                emb_dists_fix[emb_dists_fix > self.appearance_thresh] = 1.0
                dists_fix = np.minimum(dists_fix, emb_dists_fix)

                matches_fix, u_track_fix, u_detection_fix = matching.linear_assignment(dists_fix,
                                                                                       thresh=self.same_thresh)
                if len(matches_fix) > 0:
                    diffusion_track_detections[matches[:, 1]][matches_fix[:, 0], :4] = detections[matches_fix[:, 1], :4]
            #sc end
            ###################################################################################################
            start_time=time.time()
            remain_inds = diffusion_track_detections[:, 5] > self.high_det_thresh
            inds_low = diffusion_track_detections[:, 5] > self.low_det_thresh
            inds_high = diffusion_track_detections[:, 5] < self.high_det_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = diffusion_track_detections[inds_second]
            dets = diffusion_track_detections[remain_inds]
            feature_emb = np.ones((dets.shape[0], 1))
            if dets.shape[0] != 0:
                feature_emb = self.get_feature_emb(raw_img, dets[:, :4])
                trust = (dets[:, 5] - self.high_det_thresh) / (1 - self.high_det_thresh)
                af = self.alpha_fixed_emb
                dets_alpha = af + (1 - af) * (1 - trust)
            if len(dets) > 0:
                detections = [STrack(STrack.tlbr_to_tlwh(det[:4]), det[5],feature=feature) for (det,feature) in zip(dets,feature_emb)]
            else:
                detections = []
            unconfirmed = []
            tracked_stracks = []
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)
            strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
            if self.use_KalmanFilter:
                STrack.multi_predict(strack_pool)
            else:
                STrack.multi_predict_diff(strack_pool, self.predict_track_model, raw_img.shape[0],raw_img.shape[1],self.scale)
            trk_embs = [st.smooth_emb_feature for st in strack_pool]
            trk_embs = np.array(trk_embs)
            emb_cost = None if (trk_embs.shape[0] == 0 or feature_emb.shape[0] == 0) else trk_embs @ feature_emb.T
            dists = matching.iou_distance(strack_pool, detections)
            iou_matrix = 1 - dists
            #base IoU & ReID
            # emb_dists = 1 - emb_cost
            # emb_dists[dists > self.proximity_thresh] = 1.0
            #
            # dists = np.minimum(dists, emb_dists)
            # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)

            #mixed IoU & ReID
            emb_dists = 1 - emb_cost
            emb_dists[dists > self.proximity_thresh] = 1.0

            if min(dists.shape) > 0:
                iou_dists = (dists<0.9).astype(np.int32)
                if iou_dists.sum(1).max() == 1 and iou_dists.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(iou_dists), axis=1)
                    u_track = np.empty(shape=(0, 2))
                    u_detection = np.empty(shape=(0, 2))
                else:
                    iou_thresh=0.6
                    emb_thresh=0.7
                    mix_dists = matching.mixed_iou_reid(dists,emb_dists,iou_thresh,emb_thresh,self.dynamic_factor)
                    matched_indices, u_track, u_detection = matching.linear_assignment(mix_dists, thresh=self.same_thresh)
            else:
                matched_indices = np.empty(shape=(0, 2))
                u_track = np.empty(shape=(0, 2))
                u_detection = np.empty(shape=(0, 2))
            matches=matched_indices
            #end mixed IoU & ReID

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                alp = dets_alpha[idet]
                if track.state == TrackState.Tracked:
                    if self.use_KalmanFilter:
                        track.update(det,self.kalman_filter, self.frame_id)
                        track.update_feature(det.smooth_emb_feature,alp)
                    else:
                        track.update(det,None,self.frame_id)
                        track.update_feature(det.smooth_emb_feature, alp)
                    activated_starcks.append(track)
                else:
                    if self.use_KalmanFilter:
                        track.re_activate(det,self.kalman_filter, self.frame_id, new_id=False)
                    else:
                        track.re_activate(det,None, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            ##### det second
            if dets_second.shape[0] != 0:
                feature_emb_second = self.get_feature_emb(raw_img, dets_second[:, :4])
                trust = (dets_second[:, 5] - self.high_det_thresh) / (1 - self.high_det_thresh)
                af = self.alpha_fixed_emb
                dets_alpha_second = af + (1 - af) * (1 - trust)
            if len(dets_second) > 0:
                '''Detections'''
                detections_second = [STrack(STrack.tlbr_to_tlwh(det[:4]), det[5], feature=feature) for
                              (det, feature) in zip(dets_second, feature_emb_second)]
            else:
                detections_second = []

            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                alp = dets_alpha_second[idet]
                if track.state == TrackState.Tracked:
                    if self.use_KalmanFilter:
                        track.update(det,self.kalman_filter, self.frame_id)
                        track.update_feature(det.smooth_emb_feature,alp)
                    else:
                        track.update(det,None,self.frame_id,alp)
                        track.update_feature(det.smooth_emb_feature, alp)
                    activated_starcks.append(track)
                else:
                    if self.use_KalmanFilter:
                        track.re_activate(det,self.kalman_filter, self.frame_id, new_id=False)
                    else:
                        track.re_activate(det,None, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                alp = dets_alpha[idet]
                if self.use_KalmanFilter:
                    unconfirmed[itracked].update(detections[idet], self.kalman_filter, self.frame_id)
                    unconfirmed[itracked].update_feature(detections[idet].smooth_emb_feature, alp)
                else:
                    unconfirmed[itracked].update(detections[idet], None, self.frame_id)
                    unconfirmed[itracked].update_feature(detections[idet].smooth_emb_feature, alp)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
            """ Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.high_det_thresh:
                    continue
                if self.use_KalmanFilter:
                    track.activate(self.kalman_filter,self.frame_id)
                else:
                    track.activate(None, self.frame_id)
                activated_starcks.append(track)
            """ Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.pre_features=cur_features
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks,association_time+time.time()-start_time

    def scale_box(self,boxes,scale):
        return boxes/scale

    def compute_dynamic_appearance(self,dets,type):

        if type=="det":
            parameter=self.det_thresh
        elif type=="conf":
            parameter=self.association_thresh
        trust = (dets- parameter) / (1 - parameter)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)
        return dets_alpha

    def get_feature_emb(self,raw_img,boxes):
        boxes = boxes / self.scale
        feature_emb=self.encoder.inference(raw_img, boxes)
        return feature_emb

    def extract_feature(self,cur_image):
        fpn_outs=self.backbone(cur_image)
        cur_features=[]
        for proj,l_feat in zip(self.feature_projs,fpn_outs):
            cur_features.append(proj(l_feat))
        mate_info=(cur_image.shape,cur_image.device,cur_image.dtype)
        return cur_features,mate_info

    def extract_mean_track_t(self,pre_box,cur_box):
        # "xyxy"
        pre_box=xyxy2cxcywh(pre_box)
        cur_box=xyxy2cxcywh(cur_box)
        abs_box=np.abs(pre_box-cur_box)
        abs_percent=np.sum(abs_box/(pre_box+1e-5),axis=1)/4
        track_t=np.mean(abs_percent)
        return min(max(int(track_t*1000),1),999)
        # t = max(999, min(0, 1000 · f (x)  ))
        #return max(min(int(track_t*1000),1),999)
    def diffusion_postprocess(self,diffusion_outputs,conf_scores,nms_thre=0.7,conf_thre=0.6):
        pre_prediction,cur_prediction=diffusion_outputs.split(len(diffusion_outputs)//2,dim=0)
        output = [None for _ in range(len(pre_prediction))]
        for i,(pre_image_pred,cur_image_pred,association_score) in enumerate(zip(pre_prediction,cur_prediction,conf_scores)):
            association_score=association_score.flatten()
            # If none are remaining => process next image
            if not pre_image_pred.size(0):
                continue
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections=torch.zeros((2,len(cur_image_pred),7),dtype=cur_image_pred.dtype,device=cur_image_pred.device)
            detections[0,:,:4]=pre_image_pred[:,:4]
            detections[1,:,:4]=cur_image_pred[:,:4]
            detections[0,:,4]=association_score
            detections[1,:,4]=association_score
            detections[0,:,5]=torch.sqrt(torch.sigmoid(pre_image_pred[:,4])*association_score)
            detections[1,:,5]=torch.sqrt(torch.sigmoid(cur_image_pred[:,4])*association_score)
            score_out_index=association_score>conf_thre
            detections=detections[:,score_out_index,:]
            if not detections.size(1):
                output[i]=detections
                continue
            nms_out_index_3d = cluster_nms(
                                        detections[0,:,:4],
                                        detections[1,:,:4],
                                        detections[0,:,4],
                                        iou_threshold=nms_thre)
            detections = detections[:,nms_out_index_3d,:]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output[0][0],output[0][1],torch.cat([output[1][0],output[1][1]],dim=0) if len(output)>=2 else None

    def diffusion_track_filt(self,ref_detections,track_detections,conf_thre=0.6,nms_thre=0.7):
        if not ref_detections.size(1):
            return ref_detections.cpu().numpy(),track_detections.cpu().numpy()
        scores=ref_detections[:,5]
        score_out_index=scores>conf_thre
        ref_detections=ref_detections[score_out_index]
        track_detections=track_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
                ref_detections[:, :4],
                ref_detections[:, 5],
                ref_detections[:, 6],
                nms_thre,
            )
        return ref_detections[nms_out_index].cpu().numpy(),track_detections[nms_out_index].cpu().numpy()

    def diffusion_det_filt(self,diffusion_detections,conf_thre=0.6,nms_thre=0.7):
        if not diffusion_detections.size(1):
            return diffusion_detections.cpu().numpy()
        #(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        scores=diffusion_detections[:,5]
        score_out_index=scores>conf_thre
        diffusion_detections=diffusion_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
                diffusion_detections[:, :4],
                diffusion_detections[:, 5],
                diffusion_detections[:, 6],
                nms_thre,
            )
        return diffusion_detections[nms_out_index].cpu().numpy()
    
    def proposal_schedule(self,num_ref_bboxes):
        # simple strategy
        return 16*num_ref_bboxes

    def sampling_steps_schedule(self,num_ref_bboxes):
        min_sampling_steps=1
        max_sampling_steps=4
        min_num_bboxes=10
        max_num_bboxes=100
        ref_sampling_steps=(num_ref_bboxes-min_num_bboxes)*(max_sampling_steps-min_sampling_steps)/(max_num_bboxes-min_num_bboxes)+min_sampling_steps
        return min(max(int(ref_sampling_steps),min_sampling_steps),max_sampling_steps)

    def vote_to_remove_candidate(self,track_ids,detections,vote_iou_thres=0.75,sorted=False,descending=False):
        box_pred_per_image, scores_per_image=detections[:,:4],detections[:,4]*detections[:,5]
        score_track_indices=torch.argsort((track_ids+scores_per_image),descending=True)
        track_ids=track_ids[score_track_indices]
        scores_per_image=scores_per_image[score_track_indices]
        box_pred_per_image=box_pred_per_image[score_track_indices]
        assert len(track_ids)==box_pred_per_image.shape[0]
        # vote guarantee only one track id in track candidates
        keep_mask = torch.zeros_like(scores_per_image, dtype=torch.bool)
        for class_id in torch.unique(track_ids):
            curr_indices = torch.where(track_ids == class_id)[0]
            curr_keep_indices = nms(box_pred_per_image[curr_indices],scores_per_image[curr_indices],vote_iou_thres)
            candidate_iou_indices=box_iou(box_pred_per_image[curr_indices],box_pred_per_image[curr_indices])>vote_iou_thres
            counter=[]
            for cluster_indice in candidate_iou_indices[curr_keep_indices]:
                cluster_scores=scores_per_image[curr_indices][cluster_indice]
                counter.append(len(cluster_scores)+torch.mean(cluster_scores))
            max_indice=torch.argmax(torch.tensor(counter).type(self.data_type))
            keep_mask[curr_indices[curr_keep_indices][max_indice]] = True
        keep_indices = torch.where(keep_mask)[0]        
        track_ids=track_ids[keep_indices]
        box_pred_per_image=box_pred_per_image[keep_indices]
        scores_per_image=scores_per_image[keep_indices]
        if sorted and not descending:
            descending_indices=torch.argsort(track_ids)
            track_ids=track_ids[descending_indices]
            box_pred_per_image=box_pred_per_image[descending_indices]
            scores_per_image=scores_per_image[descending_indices]

        return track_ids.cpu().numpy(),box_pred_per_image.cpu().numpy(),scores_per_image.cpu().numpy()

    def prepare_input(self,pre_features,cur_features):
        inps_pre_features=[]
        inps_cur_Features=[]
        for l_pre_feat,l_cur_feat in zip(pre_features,cur_features):
            inps_pre_features.append(torch.cat([l_pre_feat.clone(),l_cur_feat.clone()],dim=0))
            inps_cur_Features.append(torch.cat([l_cur_feat.clone(),l_cur_feat.clone()],dim=0))
        return (inps_pre_features,inps_cur_Features)

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if stracksa[p].mean is not None and stracksb[q].mean is not None:
            x,y=stracksa[p].mean[4:6],stracksa[p].mean[4:6]
            cosine_dist=1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-06)
            if cosine_dist>0.15:
                continue
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb



