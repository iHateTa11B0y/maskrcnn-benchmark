import json
import numpy as np
from cython_bbox import bbox_overlaps
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
import scipy.sparse
from cvtools import cv_load_image
from colors import get_color

def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()

class coco(object):

    def __init__(self, ann_file):
        self._ann_file = ann_file
        self._COCO = COCO(self._ann_file)

        cats = self._COCO.loadCats(self._COCO.getCatIds())
        print cats
        self.num_classes = len(cats) + 1

        self._image_index = self._load_image_set_index()
        self.map = dict([[v['id'], k] for k, v in enumerate(cats)])

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        # image_ids = [int(ids) for ids in image_ids]
        return image_ids

    def gt_roidb(self):
        gt_roidb = [self._load_coco_annotation(index) for index in self._image_index]
        return gt_roidb

    def get_annotation(self, index):
        return self._load_coco_annotation(index)

    def _load_coco_annotation(self, index):
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = self.map[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0
        validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        obj_info = {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
        obj_info.update(self._COCO.loadImgs(index)[0])
        obj_info['im_shape'] = np.array([obj_info['height'], obj_info['width']])
        return obj_info

def rle_mask_nms(masks, dets, thresh=0.3):
    if len(masks) == 0:
        return []
    if len(masks) == 1:
        return [0]

    all_not_crowds = [False] * len(masks)
    ious = mask_util.iou(masks, masks, all_not_crowds)

    scores = dets[:, 4]
    order = np.argsort(-scores)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = ious[i, order[1:]]
        inds_to_keep = np.where(ovr <= thresh)[0]
        order = order[inds_to_keep + 1]

    return keep

def mask_nms_un(masks, dets, thresh=0.5):
    if len(masks) == 0:
        return []
    if len(masks) == 1:
        return [0]

    all_crowds = [True] * len(masks)
    ious = mask_util.iou(masks, masks, all_crowds)
    # ious = np.maximum(ious, ious.transpose())

    scores = dets[:, 4]
    order = np.argsort(scores)

    keep = []
    for i in order:
        if (ious[i].sum() - 1 > thresh and np.sort(ious[i])[-3] > 0.05):# or np.sort(ious[i])[-2] > 0.7:
            ious[:, i] = 0
            continue
        keep.append(i)
    
    keep2 = []
    for i in keep:
        if np.sort(ious[i])[-2] > 0.7:
            ious[:, i] = 0
            continue
        keep2.append(i)

    return keep2

def load_result(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    print "Total detection results: {}" . format(len(data))

    dt = {}
    for k, d in enumerate(data):
        if d['category_id'] > 1:
            continue
        tmp = dt.get(d['image_id'], [])
        tmp.append(d['bbox'] + [d['score']] + [k])
        dt[d['image_id']] = tmp

    print "Total images: {}" . format(len(dt))

    for d in dt:
        dt[d] = np.array(dt.pop(d), dtype=np.float)
        dt[d][:, 2] = dt[d][:, 2] + dt[d][:, 0]
        dt[d][:, 3] = dt[d][:, 3] + dt[d][:, 1]

    print dt.values()[0].shape

    return dt

def load_seg(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    return data

#### niu
def load_proposals(prop_file):
    if not prop_file:
        return None, None
    with open(prop_file, 'r') as f:
        data =json.load(f)
        
    return data[0], data[1]

def dist(p):
    cx = (p[2]+p[0]) / 2 
    cy = (p[3]+p[1]) / 2
    return cx + ( cx**2 + cy**2 )**0.5

def reorder_props_by_dist(props):
    info = [(dist(p),p) for p in props]
    info.sort(key=lambda x: x[0])
    return [x[1] for x in info]

def get_mis_gt(bl1, bl2, overlap):
    # we need len(bl1) > len(bl2) && len(overlap[0]) < len(overlap)
    max_overlap_index_in_bl2  = []
    for id, o in enumerate(overlap):
        if id>0 and np.argmax(o) == max_overlap_index_in_bl2[-1]:
            return np.argmax(o)
        max_overlap_index_in_bl2.append(np.argmax(o))
    
    return -1
        
def is_overlap(bbox1,bbox2):
    return is_in_bbox(bbox1[0],bbox1[1],bbox2) or is_in_bbox(bbox1[0],bbox1[3],bbox2) or is_in_bbox(bbox1[2],bbox1[1],bbox2) or is_in_bbox(bbox1[2],bbox1[3],bbox2)

def is_in_bbox(x,y,bbox):
    return x > bbox[0] and x < bbox[2] and y > bbox[1] and y < bbox[3]

def find_local_props(mis_gt,props,scores):
    pps = []
    scs = []
    for i, p in enumerate(props):
        if is_overlap(p, mis_gt):
            pps.append(p)
            scs.append(scores[i])
    return pps,scs

def Area(a):
    return (a[2] - a[0]) * (a[3] - a[1])
def IOU(a,b):
    if (a[0]>b[2] and a[1]>b[3]) or (b[0]>a[2] and b[1]>a[3]):
        return 0
    else:
        I = min([Area([a[0],a[1],b[2],b[3]]), Area([b[0],b[1],a[2],a[3]])])
        U = Area(a) + Area(b) - I
        return float(I) / U

   
####

def display(
        im, 
        gt_box, 
        dt_box, 
        dt_seg=None, 
        alpha=0.8, 
        save=None, 
        props=None, 
        scores=None, 
        mis_gt=None, 
        bboxes=None, 
        bbscores=None
    ):
    dt_im = im.copy()
    colors = get_color(step=11, regc=True)
    dt_mask = im.copy()
    for b in dt_box:
        c = next(colors)
        if dt_seg:
            m = mask_util.decode(dt_seg[int(b[5])]['segmentation'])
            r = cv2.cvtColor((1 - m), cv2.COLOR_GRAY2BGR) * dt_mask
            r = r + cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) * c
#            dt_mask[np.nonzero(r)] = r[np.nonzero(r)]
            cv2.addWeighted(r, 0.7, dt_mask, 0.3, 0, dt_mask)
#            cv2.imshow('test', dt_mask)
#            cv2.waitKey()

        cv2.rectangle(dt_mask, (int(b[0]), int(b[1])),(int(b[2]), int(b[3])), c.tolist(), 2)

    if dt_seg:
#        rs = [dt_seg[int(_)]['segmentation'] for _ in dt_box[:, 5]]
#        m = mask_util.decode(mask_util.merge(rs))
#        r = cv2.cvtColor((1 - m), cv2.COLOR_GRAY2BGR) * dt_im
#        r = r + cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) * np.array([255,0,0], dtype=np.uint8)
        cv2.addWeighted(dt_mask, alpha, dt_im, 1 - alpha, 0, dt_im)
    
    if save:
        cv2.imwrite(str(save) + '_pred.png',dt_im)
    else:
        cv2.imshow('Predict', dt_im)
    
    gt_im = im.copy()
    for b in gt_box:
        cv2.rectangle(gt_im, (int(b[0]), int(b[1])),(int(b[2]), int(b[3])), 65280, 5)
    
    if save:
        cv2.imwrite(str(save) + '_gt.png',gt_im)
    else:
        cv2.imshow('Ground Truth', gt_im)
        cv2.waitKey()
        #### niu
        thr = 0.1
        font = cv2.FONT_HERSHEY_SIMPLEX
        props_limit = 1000
        ps_im = im.copy()
        bb_im = im.copy()
        pl_im = im.copy()
        h,w,_ = im.shape

        #display props
        if props:
            passf = True
            for idx, p in enumerate(props):
                text = "#{}: {:.4f}".format(idx,scores[idx])
                #print(text)
                #print((int(p[0]*h), int(p[1]*w)),(int(p[2]*h), int(p[3]*w)))
                cv2.rectangle(ps_im, (int(p[0]*h), int(p[1]*w)),(int(p[2]*h), int(p[3]*w)), (0,0,0), 3)
                cv2.putText(ps_im, text, (int(p[0]*h), int(p[1]*w)), font, 0.4, (255, 255, 255), 1)
                cv2.imshow('Proposals_by_score', ps_im)
       
                if passf:
                    ke = cv2.waitKey()
                    #print(ke)
                    if int(ke) == 110:     #110 -> 'n'
                        passf = False
                    else:
                        pass
                else:
                    if p == props[-1] or (idx>0 and scores[idx]<thr and scores[idx-1]>=thr):
                        cv2.waitKey()

        if bboxes:
            passf = True
            for idx, b in enumerate(bboxes):
                #if bbscores[idx]<0.05:
                #    continue
                text = "#{}: {:.4f}".format(idx,bbscores[idx])
                #print(text)
                #print((int(b[0]*h), int(b[1]*w)),(int(b[2]*h)), int(b[3]*w))
                cv2.rectangle(bb_im, (int(b[0]*h), int(b[1]*w)),(int(b[2]*h), int(b[3]*w)), (0,0,0), 3)
                cv2.putText(bb_im, text, (int(b[0]*h), int(b[1]*w)), font, 0.4, (255, 255, 255), 1)
                cv2.imshow('bboxes_by_score', bb_im)
       
                if passf:
                    ke = cv2.waitKey()
                    #print(ke)
                    if int(ke) == 110:     #110 -> 'n'
                        passf = False
                    else:
                        pass
                else:
                    if b == bboxes[-1]: #or (idx>0 and bbscores[idx]<thr and bbscores[idx-1]>=thr):
                        cv2.waitKey()
        if mis_gt:
            #propsl = reorder_props_by_dist(props)
            propsl = []
            if mis_gt:
                mis_gt[0] /= float(h)
                mis_gt[1] /= float(w)
                mis_gt[2] /= float(h)
                mis_gt[3] /= float(w)
                propsl, scsl = find_local_props(mis_gt, props, scores)
            passf = True
            thresh_iou = 0.5
            target_iou = 0.7
            target_prop = None
            for idx, p in enumerate(propsl):
                if target_prop is None:
                    if IOU(mis_gt,p) > target_iou:
                        target_prop = p
                    else:
                        continue
                #print(IOU(mis_gt,p))
                if IOU(target_prop,p) < thresh_iou:
                    continue
                text = "#{}: {:.4f}".format(idx,scsl[idx])
                cv2.rectangle(pl_im, (int(p[0]*h), int(p[1]*w)),(int(p[2]*h), int(p[3]*w)), (0,0,0), 3)
                cv2.putText(pl_im, text, (int(p[0]*h), int(p[1]*w)), font, 0.4, (255, 255, 255), 1)
                cv2.imshow('Proposals_by_location', pl_im)

                if passf:
                    ke = cv2.waitKey()
                    #print(ke)
                    if int(ke) == 110:     #110 -> 'n'
                        passf = False
                    else:
                        pass
                else:
                    if p == propsl[-1]:# or (idx>0 and scsl[idx]<thr and scsl[idx-1]>=thr):
                        cv2.waitKey()
        #### end
                    

def eval(
    dt, 
    cc, 
    thresh=0.8, 
    seg=None, 
    show=False, 
    imids=[],
    save=False, 
    props=None, 
    scores=None, 
    bboxes=None, 
    bbscores=None
    ):
    IOUs = np.arange(0.5,1,0.05)

    d_p = np.zeros((len(dt), len(IOUs)))
    d_r = np.zeros((len(dt), len(IOUs)))

    wrong = 0.
    for idx, d in enumerate(dt.keys()):
        #### niu
        if props:
            #print(type(props))
            #print(type(scores))
            #print(u'{}'.format(d))
            prop = props[u'{}'.format(d)]
            sc = scores[u'{}'.format(d)]
        else:
            prop = None
            sc = None
        if bboxes:
            bbx = bboxes[u'{}'.format(d)]
            bbsc = bbscores[u'{}'.format(d)]
        else:
            bbx = None
            bbsc = None
        ####  
        gt = cc.get_annotation(d)
        pd_bboxes = dt[d]
#        print type(pd_bboxes), pd_bboxes.shape
#        print gt['boxes']

        all_pds = np.vstack(pd_bboxes)
        all_pds = all_pds[all_pds[:, -1] > thresh, :]
        if seg:
            rs = [seg[int(_)]['segmentation'] for _ in all_pds[:, 5]]
            keep = rle_mask_nms(rs, all_pds)
            all_pds = all_pds[keep]
            rs = [seg[int(_)]['segmentation'] for _ in all_pds[:, 5]]
            keep = mask_nms_un(rs, all_pds)
            all_pds = all_pds[keep]
        #print(len(all_pds))

        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_pds[:, :4], dtype=np.float32),
            np.ascontiguousarray(gt['boxes'], dtype=np.float32)
            )
        #print(len(overlaps))
        #print(len(overlaps[0]))
        #print(len(all_pds[:, :4]))
        #print(len(gt['boxes']))
        #print(overlaps[0])
        
#        print overlaps
        #print(overlaps.shape)
        #print(np.where(overlaps > 0.5))
        #print(np.where(overlaps > 0.5)[1])
        #raise
        if all_pds.shape[0] != gt['boxes'].shape[0]: #np.unique(np.where(overlaps > 0.5)[1]).shape[0] / float(gt['boxes'].shape[0]) != 1:
            print('olp>thr: {} | preds: {} | gts: {}'.format(np.unique(np.where(overlaps > 0.5)[1]).shape[0], all_pds.shape[0], gt['boxes'].shape[0]))
            missing = list(set(range(gt['boxes'].shape[0])) - set(np.where(overlaps > 0.5)[1]))
            more = list(set(range(all_pds.shape[0])) - set(np.where(overlaps[:, np.count_nonzero(overlaps > 0.5, axis=0)>0] > 0.5) [0]))
#            more = np.where(overlaps <= 0.5)[0]
#            print np.count_nonzero(overlaps > 0.5, axis=0), (overlaps>0.5).shape
#            raise
#            print more
            
            #### niu
            if len(overlaps) < len(overlaps[0]):
                mis_ind = get_mis_gt(gt['boxes'], all_pds[:, :4] ,np.transpose(np.array(overlaps)))
                mis_gt = all_pds[:, :4][mis_ind].tolist() if mis_ind > 0 else None
            else:
                mis_ind = get_mis_gt(all_pds[:, :4],gt['boxes'],overlaps)
                mis_gt = gt['boxes'][mis_ind].tolist() if mis_ind > 0 else None
            print(mis_ind)
            #raise
            #### end
 
            if show or save:
                print d
                idf = d if save else None
                print(gt['file_name'])
                if (len(imids)>0 and d in imids) or len(imids)==0:
                    pass
                else:
                    continue
                    
                display(
                    cv_load_image(gt['file_name']), 
                    all_pds[more], 
                    all_pds, 
                    dt_seg=seg, 
                    save=idf, 
                    props=prop, 
                    scores=sc, 
                    mis_gt=None, 
                    bboxes=bbx, 
                    bbscores=bbsc
                    )
            wrong += 1
        get_tp = lambda t: np.unique(np.where(overlaps > t)[1]).shape[0]
        vfunc = np.vectorize(get_tp)
        d_p[idx] = vfunc(IOUs) / float(all_pds.shape[0])
        d_r[idx] = vfunc(IOUs) / float(gt['boxes'].shape[0])
#        break

#    print d_p
    print "MAP:", np.mean(d_p, axis = 0)
    print "MAR:", np.mean(d_r, axis = 0)

    print "Acc by images: {}, wrong: {}, total: {}" . format(1. - wrong / len(dt), wrong, len(dt))

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument('--dt', default='', type=str,
                    help='detection result json')
    parser.add_argument('--gt', default='', type=str,
                    help='ground truth json')
    parser.add_argument('--seg', default='', type=str,
                    help='segmentation result json')
    parser.add_argument('--show', action='store_true',
                    help='show bad case')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--props', default='', type=str,
                    help='propsals from rpn')
    parser.add_argument('--bboxes', default='', type=str,
                    help='bboxes from box_head')
    parser.add_argument('--imids', default='', type=str,
                    help='specify imids to show')
    args = parser.parse_args()
    dt = load_result(args.dt)
    cc = coco(args.gt)
    seg = load_seg(args.seg) if args.seg else args.seg
    
    props, scores = load_proposals(args.props)
    bboxes,bbscores = load_proposals(args.bboxes)
    #print(bboxes[bboxes.keys()[0]])
    #print(bbscores[bbscores.keys()[0]])
    #raise
    if args.imids:
        imids = list(map(int,args.imids.split(',')))
    else:
        imids = []

    eval(
         dt, 
         cc, 
         seg=seg, 
         show=args.show, 
         imids=imids,
         save=args.save, 
         props=props, 
         scores=scores, 
         bboxes=bboxes, 
         bbscores=bbscores
        )

