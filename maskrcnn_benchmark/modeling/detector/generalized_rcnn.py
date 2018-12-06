# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        
        #### niu
        self.proposal_info = None
        self.proposal_score = None
       
        self.bboxlist = None
        self.bboxlist_score = None
        #### end

    def forward(self, images, targets=None, im_ids=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        #print('images_len: {}/{}'.format(images.tensors.shape,len(images.image_sizes)))
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        inds = [box.get_field("objectness").sort(descending=True)[1] for box in proposals]
        boxes = [box[ind] for box, ind in zip(proposals, inds)]
        scores = [b.get_field("objectness") for b in boxes]
        
        #### niu
        if im_ids and not self.training:
            self.proposal_info = {} if self.proposal_info is None else self.proposal_info
            self.proposal_score = {} if self.proposal_score is None else self.proposal_score
            print(im_ids)
            #raise
            for i, id in enumerate(list(im_ids)):
                prop = boxes[i].bbox#[:50]
                sc = scores[i]
                prop[:,0] = prop[:,0] / boxes[i].size[1]
                prop[:,1] = prop[:,1] / boxes[i].size[0]
                prop[:,2] = prop[:,2] / boxes[i].size[1]
                prop[:,3] = prop[:,3] / boxes[i].size[0]
                self.proposal_info[int(id)] = prop.cpu().numpy().tolist()
                self.proposal_score[int(id)] = sc.cpu().numpy().tolist()
            
        #### end
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        #### niu
        print('----------------------roi_head_end----------------------')
        print('imids: {}'.format(im_ids))
        if im_ids:
            print(self.training)
        if im_ids and not self.training:
            self.bboxlist = {} if self.bboxlist is None else self.bboxlist
            self.bboxlist_score = {} if self.bboxlist_score is None else self.bboxlist_score
            bboxlist = self.roi_heads.box.post_processor.bboxlist
            #bboxlist = [r.to(torch.device("cpu")) for r in result]
            import numpy as np
            for i, id in enumerate(list(im_ids)):
                #print('max socre: {}'.format(self.bboxlist[i].get_field("scores")))
                
                #data = [(bboxlist[i].bbox[_], bboxlist[i].get_field("scores")[_]) for _ in range(len(bboxlist[i].bbox))]
                #data.sort(key=lambda x: x[1],reverse = True)
                #bbox = torch.tensor(list(map(lambda x: x[0].cpu().numpy().tolist(), data)))
                #score = torch.tensor(list(map(lambda x: x[1].item(), data)))
                bbox = bboxlist[i].bbox.cpu()
                score = bboxlist[i].get_field("scores").cpu()
                #indx = np.argsort(-score.numpy())
                #bbox = torch.tensor(bbox.numpy()[indx])
                #score = torch.tensor(score.numpy()[indx])
                #print(score.shape)
                #print(indx)
                
                print(np.argmax((score.numpy())))
                print(bbox[np.argmax((score.numpy()))])
                #print(bbox[0])
                
                bbox[:,0] = bbox[:,0] / bboxlist[i].size[1]
                bbox[:,1] = bbox[:,1] / bboxlist[i].size[0]
                bbox[:,2] = bbox[:,2] / bboxlist[i].size[1]
                bbox[:,3] = bbox[:,3] / bboxlist[i].size[0]
                print(bbox[0])

                self.bboxlist[int(id)] = bbox.cpu().numpy().tolist()
                self.bboxlist_score[int(id)] = score.numpy().tolist()
                print(len(score))
                print(len(bbox))

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        
        return result
