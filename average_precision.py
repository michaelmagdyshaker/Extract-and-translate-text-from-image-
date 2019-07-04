import numpy as np

from collections import defaultdict ,namedtuple


Size    = namedtuple('Size',    ['w', 'h'])
IMG_SIZE = Size(1000, 1000)

def prop2abs(center, size, imgsize):
 
    width2  = size.w*imgsize.w/2
    height2 = size.h*imgsize.h/2
    cx      = center.x*imgsize.w
    cy      = center.y*imgsize.h
    return int(cx-width2), int(cx+width2), int(cy-height2), int(cy+height2)
#-------------------------------------------------------------------------------
def jaccard_overlap(box_arr, anchors_arr):
    areaa = (anchors_arr[:, 1]-anchors_arr[:, 0]+1) * \
            (anchors_arr[:, 3]-anchors_arr[:, 2]+1)
    areab = (box_arr[1]-box_arr[0]+1) * (box_arr[3]-box_arr[2]+1)

    xxmin = np.maximum(box_arr[0], anchors_arr[:, 0])
    xxmax = np.minimum(box_arr[1], anchors_arr[:, 1])
    yymin = np.maximum(box_arr[2], anchors_arr[:, 2])
    yymax = np.minimum(box_arr[3], anchors_arr[:, 3])

    w = np.maximum(0, xxmax-xxmin+1)
    h = np.maximum(0, yymax-yymin+1)
    intersection = w*h
    union = areab+areaa-intersection
    return intersection/union

def APs2mAP(aps):
  
    num_classes = 0.
    sum_ap = 0.
    for _, v in aps.items():
        sum_ap += v
        num_classes += 1

    if num_classes == 0:
        return 0
    return sum_ap/num_classes

#-------------------------------------------------------------------------------
class APCalculator:
   
    #---------------------------------------------------------------------------
    def __init__(self, minoverlap=0.5):
        """
        Initialize the calculator.
        """
        self.minoverlap = minoverlap
        self.clear()

    #---------------------------------------------------------------------------
    def add_detections(self, gt_boxes, boxes):
       

        sample_id = len(self.gt_boxes)
        self.gt_boxes.append(gt_boxes)

        for conf, box in boxes:
            arr = np.array(prop2abs(box.center, box.size, IMG_SIZE))
            self.det_params[box.label].append(arr)
            self.det_confidence[box.label].append(conf)
            self.det_sample_ids[box.label].append(sample_id)

    #---------------------------------------------------------------------------
    def compute_aps(self):

        #-----------------------------------------------------------------------
        # Split the ground truth samples by class and sample
        #-----------------------------------------------------------------------
        counts = defaultdict(lambda: 0)
        gt_map = defaultdict(dict)

        for sample_id, boxes in enumerate(self.gt_boxes):
            boxes_by_class = defaultdict(list)
            for box in boxes:
                counts[box.label] += 1
                boxes_by_class[box.label].append(box)

            for k, v in boxes_by_class.items():
                arr = np.zeros((len(v), 4))
                match = np.zeros((len(v)), dtype=np.bool)
                for i, box in enumerate(v):
                    arr[i] = np.array(prop2abs(box.center, box.size, IMG_SIZE))
                gt_map[k][sample_id] = (arr, match)

        #-----------------------------------------------------------------------
        # Compare predictions to ground truth
        #-----------------------------------------------------------------------
        aps = {}
        for k in gt_map:
            #-------------------------------------------------------------------
            # Create numpy arrays of detection parameters and sort them
            # in descending order
            #-------------------------------------------------------------------
            params = np.array(self.det_params[k], dtype=np.float32)
            confs = np.array(self.det_confidence[k], dtype=np.float32)
            sample_ids = np.array(self.det_sample_ids[k], dtype=np.int)
            idxs_max = np.argsort(-confs)
            params = params[idxs_max]
            confs = confs[idxs_max]
            sample_ids = sample_ids[idxs_max]

            #-------------------------------------------------------------------
            # Loop over the detections and count true and false positives
            #-------------------------------------------------------------------
            tps = np.zeros((params.shape[0])) # true positives
            fps = np.zeros((params.shape[0])) # false positives
            for i in range(params.shape[0]):
                sample_id = sample_ids[i]
                box = params[i]

                #---------------------------------------------------------------
                # The image this detection comes from contains no objects of
                # of this class
                #---------------------------------------------------------------
                if not sample_id in gt_map[k]:
                    fps[i] = 1
                    continue

                #---------------------------------------------------------------
                # Compute the jaccard overlap and see if it's over the threshold
                #---------------------------------------------------------------
                gt = gt_map[k][sample_id][0]
                matched = gt_map[k][sample_id][1]

                iou = jaccard_overlap(box, gt)
                max_idx = np.argmax(iou)

                if iou[max_idx] < self.minoverlap:
                    fps[i] = 1
                    continue

                #---------------------------------------------------------------
                # Check if the max overlap ground truth box is already matched
                #---------------------------------------------------------------
                if matched[max_idx]:
                    fps[i] = 1
                    continue

                tps[i] = 1
                matched[max_idx] = True

            #-------------------------------------------------------------------
            # Compute the precision, recall
            #-------------------------------------------------------------------
            fps = np.cumsum(fps)
            tps = np.cumsum(tps)
            recall = tps/counts[k]
            prec = tps/(tps+fps)
            ap = 0
            for r_tilde in np.arange(0, 1.1, 0.1):
                prec_rec = prec[recall>=r_tilde]
                if len(prec_rec) > 0:
                    ap += np.amax(prec_rec)

            ap /= 11.
            aps[k] = ap

        return aps

    #---------------------------------------------------------------------------
    def clear(self):

        self.det_params = defaultdict(list)
        self.det_confidence = defaultdict(list)
        self.det_sample_ids = defaultdict(list)
        self.gt_boxes = []


