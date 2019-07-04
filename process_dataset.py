import argparse
import pickle
import sys
import cv2 as cv
import os
from collections import namedtuple, defaultdict
from math import sqrt, log, exp
import random
import argparse
import numpy as np

#from transforms import *
from SSDtrain import *
#from ssdutils import get_preset_by_name
#from utils import load_data_source, str2bool, draw_box
from tqdm import tqdm

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)



#-------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------- utlis ----------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])

#-------------------------------------------------------------------------------
Label   = namedtuple('Label',   ['name', 'color'])
Size    = namedtuple('Size',    ['w', 'h'])
Point   = namedtuple('Point',   ['x', 'y'])
Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize'])
Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size'])
Score   = namedtuple('Score',   ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good'])

#-------------------------------------------------------------------------------
def str2bool(v):
    """
    Convert a string to a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-------------------------------------------------------------------------------
def abs2prop(xmin, xmax, ymin, ymax, imgsize):
    """
    Convert the absolute min-max box bound to proportional center-width bounds
    """
    width   = float(xmax-xmin)
    height  = float(ymax-ymin)
    cx      = float(xmin)+width/2
    cy      = float(ymin)+height/2
    width  /= imgsize.w
    height /= imgsize.h
    cx     /= imgsize.w
    cy     /= imgsize.h
    return Point(cx, cy), Size(width, height)

#-------------------------------------------------------------------------------
def prop2abs(center, size, imgsize):
    """
    Convert proportional center-width bounds to absolute min-max bounds
    """
    width2  = size.w*imgsize.w/2
    height2 = size.h*imgsize.h/2
    cx      = center.x*imgsize.w
    cy      = center.y*imgsize.h
    return int(cx-width2), int(cx+width2), int(cy-height2), int(cy+height2)

#-------------------------------------------------------------------------------
def box_is_valid(box):
    for x in [box.center.x, box.center.y, box.size.w, box.size.h]:
        if math.isnan(x) or math.isinf(x):
            return False
    return True

#-------------------------------------------------------------------------------
def normalize_box(box):
    if not box_is_valid(box):
        return box

    img_size = Size(1000, 1000)
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    xmin = max(xmin, 0)
    xmax = min(xmax, img_size.w-1)
    ymin = max(ymin, 0)
    ymax = min(ymax, img_size.h-1)

    # this happens early in the training when box min and max are outside
    # of the image
    xmin = min(xmin, xmax)
    ymin = min(ymin, ymax)

    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size)
    return Box(box.label, box.labelid, center, size)

#-------------------------------------------------------------------------------
def draw_box(img, box, color):
    img_size = Size(img.shape[1], img.shape[0])
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    img_box = np.copy(img)
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_box, box.label, (xmin+5, ymin-5), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
    return img_size

#-------------------------------------------------------------------------------
class PrecisionSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, labels, restore=False):
        self.session = session
        self.writer = writer
        self.labels = labels

        sess = session
        ph_name = sample_name+'_mAP_ph'
        sum_name = sample_name+'_mAP'

        if restore:
            self.mAP_placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
            self.mAP_summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
        else:
            self.mAP_placeholder = tf.placeholder(tf.float32, name=ph_name)
            self.mAP_summary_op = tf.summary.scalar(sum_name,
                                                    self.mAP_placeholder)

        self.placeholders = {}
        self.summary_ops = {}

        for label in labels:
            sum_name = sample_name+'_AP_'+label
            ph_name = sample_name+'_AP_ph_'+label
            if restore:
                placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
                summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
            else:
                placeholder = tf.placeholder(tf.float32, name=ph_name)
                summary_op = tf.summary.scalar(sum_name, placeholder)
            self.placeholders[label] = placeholder
            self.summary_ops[label] = summary_op

    #---------------------------------------------------------------------------
    def push(self, epoch, mAP, APs):
        if not APs: return

        feed = {self.mAP_placeholder: mAP}
        tensors = [self.mAP_summary_op]
        for label in self.labels:
            feed[self.placeholders[label]] = APs[label]
            tensors.append(self.summary_ops[label])

        summaries = self.session.run(tensors, feed_dict=feed)

        for summary in summaries:
            self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, colors, restore=False):
        self.session = session
        self.writer = writer
        self.colors = colors

        sess = session
        sum_name = sample_name+'_img'
        ph_name = sample_name+'_img_ph'
        if restore:
            self.img_placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
            self.img_summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
        else:
            self.img_placeholder = tf.placeholder(tf.float32, name=ph_name,
                                                  shape=[None, None, None, 3])
            self.img_summary_op = tf.summary.image(sum_name,
                                                   self.img_placeholder)

    #---------------------------------------------------------------------------
    def push(self, epoch, samples):
        imgs = np.zeros((3, 512, 512, 3))
        for i, sample in enumerate(samples):
            img = cv2.resize(sample[0], (512, 512))
            for _, box in sample[1]:
                draw_box(img, box, self.colors[box.label])
            img[img>255] = 255
            img[img<0] = 0
            imgs[i] = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        feed = {self.img_placeholder: imgs}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class LossSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, num_samples,
                 restore=False):
        self.session = session
        self.writer = writer
        self.num_samples = num_samples
        self.loss_names = ['total', 'localization', 'confidence', 'l2']
        self.loss_values = {}
        self.placeholders = {}

        sess = session

        summary_ops = []
        for loss in self.loss_names:
            sum_name = sample_name+'_'+loss+'_loss'
            ph_name = sample_name+'_'+loss+'_loss_ph'

            if restore:
                placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
                summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
            else:
                placeholder = tf.placeholder(tf.float32, name=ph_name)
                summary_op = tf.summary.scalar(sum_name, placeholder)

            self.loss_values[loss] = float(0)
            self.placeholders[loss] = placeholder
            summary_ops.append(summary_op)

        self.summary_ops = tf.summary.merge(summary_ops)

    #---------------------------------------------------------------------------
    def add(self, values, num_samples):
        for loss in self.loss_names:
            self.loss_values[loss] += values[loss]*num_samples

    #---------------------------------------------------------------------------
    def push(self, epoch):
        feed = {}
        for loss in self.loss_names:
            feed[self.placeholders[loss]] = \
                self.loss_values[loss]/self.num_samples

        summary = self.session.run(self.summary_ops, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        for loss in self.loss_names:
            self.loss_values[loss] = float(0)


#-------------------------------------------------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])

#-------------------------------------------------------------------------------
Label   = namedtuple('Label',   ['name', 'color'])
Size    = namedtuple('Size',    ['w', 'h'])
Point   = namedtuple('Point',   ['x', 'y'])
Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize'])
Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size'])
Score   = namedtuple('Score',   ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good'])

#-------------------------------------------------------------------------------
def str2bool(v):
    """
    Convert a string to a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-------------------------------------------------------------------------------
def abs2prop(xmin, xmax, ymin, ymax, imgsize):
    """
    Convert the absolute min-max box bound to proportional center-width bounds
    """
    width   = float(xmax-xmin)
    height  = float(ymax-ymin)
    cx      = float(xmin)+width/2
    cy      = float(ymin)+height/2
    width  /= imgsize.w
    height /= imgsize.h
    cx     /= imgsize.w
    cy     /= imgsize.h
    return Point(cx, cy), Size(width, height)

#-------------------------------------------------------------------------------
def prop2abs(center, size, imgsize):
    """
    Convert proportional center-width bounds to absolute min-max bounds
    """
    width2  = size.w*imgsize.w/2
    height2 = size.h*imgsize.h/2
    cx      = center.x*imgsize.w
    cy      = center.y*imgsize.h
    return int(cx-width2), int(cx+width2), int(cy-height2), int(cy+height2)

#-------------------------------------------------------------------------------
def box_is_valid(box):
    for x in [box.center.x, box.center.y, box.size.w, box.size.h]:
        if math.isnan(x) or math.isinf(x):
            return False
    return True

#-------------------------------------------------------------------------------
def normalize_box(box):
    if not box_is_valid(box):
        return box

    img_size = Size(1000, 1000)
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    xmin = max(xmin, 0)
    xmax = min(xmax, img_size.w-1)
    ymin = max(ymin, 0)
    ymax = min(ymax, img_size.h-1)

    # this happens early in the training when box min and max are outside
    # of the image
    xmin = min(xmin, xmax)
    ymin = min(ymin, ymax)

    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size)
    return Box(box.label, box.labelid, center, size)

#-------------------------------------------------------------------------------
def draw_box(img, box, color):
    img_size = Size(img.shape[1], img.shape[0])
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    img_box = np.copy(img)
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_box, box.label, (xmin+5, ymin-5), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
    return img_size
def draw_box2(img, box, color):
    img_size = Size(img.shape[1], img.shape[0])
    #xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    #img_box = np.copy(img)
    #cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    #cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img_box, box.label, (xmin+5, ymin-5), font, 0.5,
    #            (255, 255, 255), 1, cv2.LINE_AA)
    #alpha = 0.8
    #cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
    return img_size
#-------------------------------------------------------------------------------
class PrecisionSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, labels, restore=False):
        self.session = session
        self.writer = writer
        self.labels = labels

        sess = session
        ph_name = sample_name+'_mAP_ph'
        sum_name = sample_name+'_mAP'

        if restore:
            self.mAP_placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
            self.mAP_summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
        else:
            self.mAP_placeholder = tf.placeholder(tf.float32, name=ph_name)
            self.mAP_summary_op = tf.summary.scalar(sum_name,
                                                    self.mAP_placeholder)

        self.placeholders = {}
        self.summary_ops = {}

        for label in labels:
            sum_name = sample_name+'_AP_'+label
            ph_name = sample_name+'_AP_ph_'+label
            if restore:
                placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
                summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
            else:
                placeholder = tf.placeholder(tf.float32, name=ph_name)
                summary_op = tf.summary.scalar(sum_name, placeholder)
            self.placeholders[label] = placeholder
            self.summary_ops[label] = summary_op

    #---------------------------------------------------------------------------
    def push(self, epoch, mAP, APs):
        if not APs: return

        feed = {self.mAP_placeholder: mAP}
        tensors = [self.mAP_summary_op]
        for label in self.labels:
            feed[self.placeholders[label]] = APs[label]
            tensors.append(self.summary_ops[label])

        summaries = self.session.run(tensors, feed_dict=feed)

        for summary in summaries:
            self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, colors, restore=False):
        self.session = session
        self.writer = writer
        self.colors = colors

        sess = session
        sum_name = sample_name+'_img'
        ph_name = sample_name+'_img_ph'
        if restore:
            self.img_placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
            self.img_summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
        else:
            self.img_placeholder = tf.placeholder(tf.float32, name=ph_name,
                                                  shape=[None, None, None, 3])
            self.img_summary_op = tf.summary.image(sum_name,
                                                   self.img_placeholder)

    #---------------------------------------------------------------------------
    def push(self, epoch, samples):
        imgs = np.zeros((3, 512, 512, 3))
        for i, sample in enumerate(samples):
            img = cv2.resize(sample[0], (512, 512))
            for _, box in sample[1]:
                draw_box(img, box, self.colors[box.label])
            img[img>255] = 255
            img[img<0] = 0
            imgs[i] = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        feed = {self.img_placeholder: imgs}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class LossSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, num_samples,
                 restore=False):
        self.session = session
        self.writer = writer
        self.num_samples = num_samples
        self.loss_names = ['total', 'localization', 'confidence', 'l2']
        self.loss_values = {}
        self.placeholders = {}

        sess = session

        summary_ops = []
        for loss in self.loss_names:
            sum_name = sample_name+'_'+loss+'_loss'
            ph_name = sample_name+'_'+loss+'_loss_ph'

            if restore:
                placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
                summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
            else:
                placeholder = tf.placeholder(tf.float32, name=ph_name)
                summary_op = tf.summary.scalar(sum_name, placeholder)

            self.loss_values[loss] = float(0)
            self.placeholders[loss] = placeholder
            summary_ops.append(summary_op)

        self.summary_ops = tf.summary.merge(summary_ops)

    #---------------------------------------------------------------------------
    def add(self, values, num_samples):
        for loss in self.loss_names:
            self.loss_values[loss] += values[loss]*num_samples

    #---------------------------------------------------------------------------
    def push(self, epoch):
        feed = {}
        for loss in self.loss_names:
            feed[self.placeholders[loss]] = \
                self.loss_values[loss]/self.num_samples

        summary = self.session.run(self.summary_ops, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        for loss in self.loss_names:
            self.loss_values[loss] = float(0)


#-------------------------------------------------------------------------------
# Define the flavors of SSD that we're going to use and it's various properties.
# It's done so that we don't have to build the whole network in memory in order
# to pre-process the datasets.
#-------------------------------------------------------------------------------
SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SSDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])

SSD_PRESETS = {
    'vgg300': SSDPreset(name = 'vgg300',
                        image_size = Size(300, 300),
                        maps = [
                            SSDMap(Size(38, 38), 0.1,   [2, 0.5]),
                            SSDMap(Size(19, 19), 0.2,   [2, 3, 0.5, 1./3.]),
                            SSDMap(Size(10, 10), 0.375, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 5,  5), 0.55,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 3,  3), 0.725, [2, 0.5]),
                            SSDMap(Size( 1,  1), 0.9,   [2, 0.5])
                        ],
                        extra_scale = 1.075,
                        num_anchors = 8732),
    'vgg512': SSDPreset(name = 'vgg512',
                        image_size = Size(512, 512),
                        maps = [
                            SSDMap(Size(64, 64), 0.07, [2, 0.5]),
                            SSDMap(Size(32, 32), 0.15, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size(16, 16), 0.3,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 8,  8), 0.45, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 4,  4), 0.6,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 2,  2), 0.75, [2, 0.5]),
                            SSDMap(Size( 1,  1), 0.9,  [2, 0.5])
                        ],
                        extra_scale = 1.05,
                        num_anchors = 24564)
}

#-------------------------------------------------------------------------------
# Default box parameters both in terms proportional to image dimensions
#-------------------------------------------------------------------------------
Anchor = namedtuple('Anchor', ['center', 'size', 'x', 'y', 'scale', 'map'])

#-------------------------------------------------------------------------------
def get_preset_by_name(pname):
    if not pname in SSD_PRESETS:
        raise RuntimeError('No such preset: '+pname)
    return SSD_PRESETS[pname]

#-------------------------------------------------------------------------------
def get_anchors_for_preset(preset):
    """
    Compute the default (anchor) boxes for the given SSD preset
    """
    #---------------------------------------------------------------------------
    # Compute the width and heights of the anchor boxes for every scale
    #---------------------------------------------------------------------------
    box_sizes = []
    for i in range(len(preset.maps)):
        map_params = preset.maps[i]
        s = map_params.scale
        aspect_ratios = [1] + map_params.aspect_ratios
        aspect_ratios = list(map(lambda x: sqrt(x), aspect_ratios))

        sizes = []
        for ratio in aspect_ratios:
            w = s * ratio
            h = s / ratio
            sizes.append((w, h))
        if i < len(preset.maps)-1:
            s_prime = sqrt(s*preset.maps[i+1].scale)
        else:
            s_prime = sqrt(s*preset.extra_scale)
        sizes.append((s_prime, s_prime))
        box_sizes.append(sizes)

    #---------------------------------------------------------------------------
    # Compute the actual boxes for every scale and feature map
    #---------------------------------------------------------------------------
    anchors = []
    for k in range(len(preset.maps)):
        fk = preset.maps[k].size[0]
        s = preset.maps[k].scale
        for size in box_sizes[k]:
            for j in range(fk):
                y = (j+0.5)/float(fk)
                for i in range(fk):
                    x = (i+0.5)/float(fk)
                    box = Anchor(Point(x, y), Size(size[0], size[1]),
                                 i, j, s, k)
                    anchors.append(box)
    return anchors

#-------------------------------------------------------------------------------
def anchors2array(anchors, img_size):
    """
    Computes a numpy array out of absolute anchor params (img_size is needed
    as a reference)
    """
    arr = np.zeros((len(anchors), 4))
    for i in range(len(anchors)):
        anchor = anchors[i]
        xmin, xmax, ymin, ymax = prop2abs(anchor.center, anchor.size, img_size)
        arr[i] = np.array([xmin, xmax, ymin, ymax])
    return arr

#-------------------------------------------------------------------------------
def box2array(box, img_size):
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    return np.array([xmin, xmax, ymin, ymax])

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

#-------------------------------------------------------------------------------
def compute_overlap(box_arr, anchors_arr, threshold):
    iou = jaccard_overlap(box_arr, anchors_arr)
    overlap = iou > threshold

    good_idxs = np.nonzero(overlap)[0]
    best_idx  = np.argmax(iou)
    best = None
    good = []

    if iou[best_idx] > threshold:
        best = Score(best_idx, iou[best_idx])

    for idx in good_idxs:
        good.append(Score(idx, iou[idx]))

    return Overlap(best, good)

#-------------------------------------------------------------------------------
def compute_location(box, anchor):
    arr = np.zeros((4))
    arr[0] = (box.center.x-anchor.center.x)/anchor.size.w*10
    arr[1] = (box.center.y-anchor.center.y)/anchor.size.h*10
    arr[2] = log(box.size.w/anchor.size.w)*5
    arr[3] = log(box.size.h/anchor.size.h)*5
    return arr

#-------------------------------------------------------------------------------
def decode_location(box, anchor):
    box[box > 100] = 100 # only happens early training

    x = box[0]/10 * anchor.size.w + anchor.center.x
    y = box[1]/10 * anchor.size.h + anchor.center.y
    w = exp(box[2]/5) * anchor.size.w
    h = exp(box[3]/5) * anchor.size.h
    return Point(x, y), Size(w, h)

#-------------------------------------------------------------------------------
def decode_boxes(pred, anchors, confidence_threshold = 0.01, lid2name = {},
                 detections_cap=200):
    """
    Decode boxes from the neural net predictions.
    Label names are decoded using the lid2name dictionary - the id to name
    translation is not done if the corresponding key does not exist.
    """

    #---------------------------------------------------------------------------
    # Find the detections
    #---------------------------------------------------------------------------
    num_classes = pred.shape[1]-4
    bg_class    = num_classes-1
    box_class   = np.argmax(pred[:, :num_classes-1], axis=1)
    confidence  = pred[np.arange(len(pred)), box_class]
    if detections_cap is not None:
        detections = np.argsort(confidence)[::-1][:detections_cap]
    else:
        detections = np.argsort(confidence)[::-1]

    #---------------------------------------------------------------------------
    # Decode coordinates of each box with confidence over a threshold
    #---------------------------------------------------------------------------
    boxes = []
    for idx in detections:
        confidence = pred[idx, box_class[idx]]
        if confidence < confidence_threshold:
            break

        center, size = decode_location(pred[idx, num_classes:], anchors[idx])
        cid = box_class[idx]
        cname = None
        if cid in lid2name:
            cname = lid2name[cid]
        det = (confidence, normalize_box(Box(cname, cid, center, size)))
        boxes.append(det)

    return boxes

#-------------------------------------------------------------------------------
def non_maximum_suppression(boxes, overlap_threshold):
    #---------------------------------------------------------------------------
    # Convert to absolute coordinates and to a more convenient format
    #---------------------------------------------------------------------------
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    conf = []
    img_size = Size(1000, 1000)

    for box in boxes:
        params = prop2abs(box[1].center, box[1].size, img_size)
        xmin.append(params[0])
        xmax.append(params[1])
        ymin.append(params[2])
        ymax.append(params[3])
        conf.append(box[0])

    xmin = np.array(xmin)
    xmax = np.array(xmax)
    ymin = np.array(ymin)
    ymax = np.array(ymax)
    conf = np.array(conf)

    #---------------------------------------------------------------------------
    # Compute the area of each box and sort the indices by confidence level
    # (lowest confidence first first).
    #---------------------------------------------------------------------------
    area = (xmax-xmin+1) * (ymax-ymin+1)
    idxs = np.argsort(conf)
    pick = []

    #---------------------------------------------------------------------------
    # Loop until we still have indices to process
    #---------------------------------------------------------------------------
    while len(idxs) > 0:
        #-----------------------------------------------------------------------
        # Grab the last index (ie. the most confident detection), remove it from
        # the list of indices to process, and put it on the list of picks
        #-----------------------------------------------------------------------
        last = idxs.shape[0]-1
        i    = idxs[last]
        idxs = np.delete(idxs, last)
        pick.append(i)
        suppress = []

        #-----------------------------------------------------------------------
        # Figure out the intersection with the remaining windows
        #-----------------------------------------------------------------------
        xxmin = np.maximum(xmin[i], xmin[idxs])
        xxmax = np.minimum(xmax[i], xmax[idxs])
        yymin = np.maximum(ymin[i], ymin[idxs])
        yymax = np.minimum(ymax[i], ymax[idxs])

        w = np.maximum(0, xxmax-xxmin+1)
        h = np.maximum(0, yymax-yymin+1)
        intersection = w*h

        #-----------------------------------------------------------------------
        # Compute IOU and suppress indices with IOU higher than a threshold
        #-----------------------------------------------------------------------
        union    = area[i]+area[idxs]-intersection
        iou      = intersection/union
        overlap  = iou > overlap_threshold
        suppress = np.nonzero(overlap)[0]
        idxs     = np.delete(idxs, suppress)

    #---------------------------------------------------------------------------
    # Return the selected boxes
    #---------------------------------------------------------------------------
    selected = []
    for i in pick:

        selected.append(boxes[i])

    return selected

#-------------------------------------------------------------------------------
def suppress_overlaps(boxes):
    class_boxes    = defaultdict(list)
    selected_boxes = []
    for box in boxes:
        class_boxes[box[1].labelid].append(box)

    for k, v in class_boxes.items():
        selected_boxes += non_maximum_suppression(v, 0.45)
    return selected_boxes

#------------------------------------------------------------------- transforms ----------------------------

class Transform:
    def __init__(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        self.initialized = False

#-------------------------------------------------------------------------------
class ImageLoaderTransform(Transform):
    """
    Load and image from the file specified in the Sample object
    """
    def __call__(self, data, label, gt):
        return cv2.imread(gt.filename), label, gt

#-------------------------------------------------------------------------------
def process_overlap(overlap, box, anchor, matches, num_classes, vec):
    if overlap.idx in matches and matches[overlap.idx] >= overlap.score:
        return

    matches[overlap.idx] = overlap.score
    vec[overlap.idx, 0:num_classes+1] = 0
    vec[overlap.idx, box.labelid]     = 1
    vec[overlap.idx, num_classes+1:]  = compute_location(box, anchor)

#-------------------------------------------------------------------------------
class LabelCreatorTransform(Transform):
    """
    Create a label vector out of a ground trut sample
    Parameters: preset, num_classes
    """
    #---------------------------------------------------------------------------
    def initialize(self):
        self.anchors = get_anchors_for_preset(self.preset)
        self.vheight = len(self.anchors)
        self.vwidth = self.num_classes+5 # background class + location offsets
        self.img_size = Size(1000, 1000)
        self.anchors_arr = anchors2array(self.anchors, self.img_size)
        self.initialized = True

    #---------------------------------------------------------------------------
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Initialize the data vector and other variables
        #-----------------------------------------------------------------------
        if not self.initialized:
            self.initialize()

        vec = np.zeros((self.vheight, self.vwidth), dtype=np.float32)

        #-----------------------------------------------------------------------
        # For every box compute the best match and all the matches above 0.5
        # Jaccard overlap
        #-----------------------------------------------------------------------
        overlaps = {}
        for box in gt.boxes:
            box_arr = box2array(box, self.img_size)
            overlaps[box] = compute_overlap(box_arr, self.anchors_arr, 0.5)

        #-----------------------------------------------------------------------
        # Set up the training vector resolving conflicts in favor of a better
        # match
        #-----------------------------------------------------------------------
        vec[:, self.num_classes]   = 1 # background class
        vec[:, self.num_classes+1] = 0 # x offset
        vec[:, self.num_classes+2] = 0 # y offset
        vec[:, self.num_classes+3] = 0 # log width scale
        vec[:, self.num_classes+4] = 0 # log height scale

        matches = {}
        for box in gt.boxes:
            for overlap in overlaps[box].good:
                anchor = self.anchors[overlap.idx]
                process_overlap(overlap, box, anchor, matches, self.num_classes, vec)

        matches = {}
        for box in gt.boxes:
            overlap = overlaps[box].best
            if not overlap:
                continue
            anchor  = self.anchors[overlap.idx]
            process_overlap(overlap, box, anchor, matches, self.num_classes, vec)

        return data, vec, gt

#-------------------------------------------------------------------------------
class ResizeTransform(Transform):
    """
    Resize an image
    Parameters: width, height, algorithms
    """
    def __call__(self, data, label, gt):
        alg = random.choice(self.algorithms)
        resized = cv2.resize(data, (self.width, self.height), interpolation=alg)
        return resized, label, gt

#-------------------------------------------------------------------------------
class RandomTransform(Transform):
    """
    Call another transform with a given probability
    Parameters: prob, transform
    """
    def __call__(self, data, label, gt):
        p = random.uniform(0, 1)
        if p < self.prob:
            return self.transform(data, label, gt)
        return data, label, gt

#-------------------------------------------------------------------------------
class ComposeTransform(Transform):
    """
    Call a bunch of transforms serially
    Parameters: transforms
    """
    def __call__(self, data, label, gt):
        args = (data, label, gt)
        for t in self.transforms:
            args = t(*args)
        return args

#-------------------------------------------------------------------------------
class TransformPickerTransform(Transform):
    """
    Call a randomly chosen transform from the list
    Parameters: transforms
    """
    def __call__(self, data, label, gt):
        pick = random.randint(0, len(self.transforms)-1)
        return self.transforms[pick](data, label, gt)

#-------------------------------------------------------------------------------
class BrightnessTransform(Transform):
    """
    Transform brightness
    Parameters: delta
    """
    def __call__(self, data, label, gt):
        data = data.astype(np.float32)
        delta = random.randint(-self.delta, self.delta)
        data += delta
        data[data>255] = 255
        data[data<0] = 0
        data = data.astype(np.uint8)
        return data, label, gt

#-------------------------------------------------------------------------------
class ContrastTransform(Transform):
    """
    Transform contrast
    Parameters: lower, upper
    """
    def __call__(self, data, label, gt):
        data = data.astype(np.float32)
        delta = random.uniform(self.lower, self.upper)
        data *= delta
        data[data>255] = 255
        data[data<0] = 0
        data = data.astype(np.uint8)
        return data, label, gt

#-------------------------------------------------------------------------------
class HueTransform(Transform):
    """
    Transform hue
    Parameters: delta
    """
    def __call__(self, data, label, gt):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        data = data.astype(np.float32)
        delta = random.randint(-self.delta, self.delta)
        data[0] += delta
        data[0][data[0]>180] -= 180
        data[0][data[0]<0] +=180
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        return data, label, gt

#-------------------------------------------------------------------------------
class SaturationTransform(Transform):
    """
    Transform hue
    Parameters: lower, upper
    """
    def __call__(self, data, label, gt):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        data = data.astype(np.float32)
        delta = random.uniform(self.lower, self.upper)
        data[1] *= delta
        data[1][data[1]>255] = 255
        data[1][data[1]<0] = 0
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        return data, label, gt

#-------------------------------------------------------------------------------
class ReorderChannelsTransform(Transform):
    """
    Reorder Image Channels
    """
    def __call__(self, data, label, gt):
        channels = [0, 1, 2]
        random.shuffle(channels)
        return data[:, :,channels], label, gt

#-------------------------------------------------------------------------------
def transform_box(box, orig_size, new_size, h_off, w_off):
    #---------------------------------------------------------------------------
    # Compute the new coordinates of the box
    #---------------------------------------------------------------------------
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, orig_size)
    xmin += w_off
    xmax += w_off
    ymin += h_off
    ymax += h_off

    #---------------------------------------------------------------------------
    # Check if the center falls within the image
    #---------------------------------------------------------------------------
    width = xmax - xmin
    height = ymax - ymin
    new_cx = xmin + int(width/2)
    new_cy = ymin + int(height/2)
    if new_cx < 0 or new_cx >= new_size.w:
        return None
    if new_cy < 0 or new_cy >= new_size.h:
        return None

    center, size = abs2prop(xmin, xmax, ymin, ymax, new_size)
    return Box(box.label, box.labelid, center, size)

#-------------------------------------------------------------------------------
def transform_gt(gt, new_size, h_off, w_off):
    boxes = []
    for box in gt.boxes:
        box = transform_box(box, gt.imgsize, new_size, h_off, w_off)
        if box is None:
            continue
        boxes.append(box)
    return Sample(gt.filename, boxes, new_size)

#-------------------------------------------------------------------------------
class ExpandTransform(Transform):
    """
    Expand the image and fill the empty space with the mean value
    Parameters: max_ratio, mean_value
    """
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Calculate sizes and offsets
        #-----------------------------------------------------------------------
        ratio = random.uniform(1, self.max_ratio)
        orig_size = gt.imgsize
        new_size = Size(int(orig_size.w*ratio), int(orig_size.h*ratio))
        h_off = random.randint(0, new_size.h-orig_size.h)
        w_off = random.randint(0, new_size.w-orig_size.w)

        #-----------------------------------------------------------------------
        # Create the new image and place the input image in it
        #-----------------------------------------------------------------------
        img = np.zeros((new_size.h, new_size.w, 3))
        img[:, :] = np.array(self.mean_value)
        img[h_off:h_off+orig_size.h, w_off:w_off+orig_size.w, :] = data

        #-----------------------------------------------------------------------
        # Transform the ground truth
        #-----------------------------------------------------------------------
        gt = transform_gt(gt, new_size, h_off, w_off)

        return img, label, gt

#-------------------------------------------------------------------------------
class SamplerTransform(Transform):
    """
    Sample a fraction of the image according to given parameters
    Params: min_scale, max_scale, min_aspect_ratio, max_aspect_ratio,
            min_jaccard_overlap
    """
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Check whether to sample or not
        #-----------------------------------------------------------------------
        if not self.sample:
            return data, label, gt

        #-----------------------------------------------------------------------
        # Retry sampling a couple of times
        #-----------------------------------------------------------------------
        source_boxes = anchors2array(gt.boxes, gt.imgsize)
        box = None
        box_arr = None
        for _ in range(self.max_trials):
            #-------------------------------------------------------------------
            # Sample a bounding box
            #-------------------------------------------------------------------
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect_ratio = random.uniform(self.min_aspect_ratio,
                                          self.max_aspect_ratio)

            # make sure width and height will not be larger than 1
            aspect_ratio = max(aspect_ratio, scale**2)
            aspect_ratio = min(aspect_ratio, 1/(scale**2))

            width = scale*sqrt(aspect_ratio)
            height = scale/sqrt(aspect_ratio)
            cx = 0.5*width + random.uniform(0, 1-width)
            cy = 0.5*height + random.uniform(0, 1-height)
            center = Point(cx, cy)
            size = Size(width, height)

            #-------------------------------------------------------------------
            # Check if the box satisfies the jaccard overlap constraint
            #-------------------------------------------------------------------
            box_arr = np.array(prop2abs(center, size, gt.imgsize))
            overlap = compute_overlap(box_arr, source_boxes, 0)
            if overlap.best and overlap.best.score >= self.min_jaccard_overlap:
                box = Box(None, None, center, size)
                break

        if box is None:
            return None

        #-----------------------------------------------------------------------
        # Crop the box and adjust the ground truth
        #-----------------------------------------------------------------------
        new_size = Size(box_arr[1]-box_arr[0], box_arr[3]-box_arr[2])
        w_off = -box_arr[0]
        h_off = -box_arr[2]
        data = data[box_arr[2]:box_arr[3], box_arr[0]:box_arr[1]]
        gt = transform_gt(gt, new_size, h_off, w_off)

        return data, label, gt

#-------------------------------------------------------------------------------
class SamplePickerTransform(Transform):
    """
    Run a bunch of sample transforms and return one of the produced samples
    Parameters: samplers
    """
    def __call__(self, data, label, gt):
        samples = []
        for sampler in self.samplers:
            sample = sampler(data, label, gt)
            if sample is not None:
                samples.append(sample)
        return random.choice(samples)

#-------------------------------------------------------------------------------
class HorizontalFlipTransform(Transform):
    """
    Horizontally flip the image
    """
    def __call__(self, data, label, gt):
        data = cv2.flip(data, 1)
        boxes = []
        for box in gt.boxes:
            center = Point(1-box.center.x, box.center.y)
            box = Box(box.label, box.labelid, center, box.size)
            boxes.append(box)
        gt = Sample(gt.filename, boxes, gt.imgsize)

        return data, label, gt




#-------------------------------------------------------------------------------
def annotate(data_dir, samples, colors, sample_name):

    result_dir = data_dir+'/annotated/'+sample_name.strip()+'/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for sample in tqdm(samples, desc=sample_name, unit='samples'):
        img    = cv2.imread(sample.filename)
        basefn = os.path.basename(sample.filename)
        for box in sample.boxes:
            draw_box(img, box, colors[box.label])
        cv2.imwrite(result_dir+basefn, img)

#-------------------------------------------------------------------------------
def build_sampler(overlap, trials):
    return SamplerTransform(sample=True, min_scale=0.3, max_scale=1.0,
                            min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                            min_jaccard_overlap=overlap, max_trials=trials)

#-------------------------------------------------------------------------------
def build_train_transforms(preset, num_classes, sampler_trials, expand_prob):
    #---------------------------------------------------------------------------
    # Resizing
    #---------------------------------------------------------------------------
    tf_resize = ResizeTransform(width=preset.image_size.w,
                                height=preset.image_size.h,
                                algorithms=[#cv2.INTER_LINEAR,
                                            #cv2.INTER_AREA,
                                            #cv2.INTER_NEAREST,
                                            cv2.INTER_CUBIC
                                            #cv2.INTER_LANCZOS4
                                            ])

    #---------------------------------------------------------------------------
    # Image distortions
    #---------------------------------------------------------------------------
    tf_brightness = BrightnessTransform(delta=20)
    tf_rnd_brightness = RandomTransform(prob=0.5, transform=tf_brightness)

    tf_contrast = ContrastTransform(lower=0.5, upper=1.5)
    tf_rnd_contrast = RandomTransform(prob=0.5, transform=tf_contrast)

    #tf_hue = HueTransform(delta=18)
    #tf_rnd_hue = RandomTransform(prob=0.5, transform=tf_hue)

    #tf_saturation = SaturationTransform(lower=0.5, upper=1.5)
    #tf_rnd_saturation = RandomTransform(prob=0.5, transform=tf_saturation)

    #tf_reorder_channels = ReorderChannelsTransform()
    #tf_rnd_reorder_channels = RandomTransform(prob=0.5,
    #                                          transform=tf_reorder_channels)

    #---------------------------------------------------------------------------
    # Compositions of image distortions
    #---------------------------------------------------------------------------
    tf_distort_lst = [
    #    tf_rnd_contrast,
    #    tf_rnd_saturation,
    #    tf_rnd_hue,
       tf_rnd_contrast
    ]
    tf_distort_1 = ComposeTransform(transforms=tf_distort_lst[:-1])
    tf_distort_2 = ComposeTransform(transforms=tf_distort_lst[1:])
    tf_distort_comp = [tf_distort_1, tf_distort_2]
    tf_distort = TransformPickerTransform(transforms=tf_distort_comp)

    #---------------------------------------------------------------------------
    # Expand sample
    #---------------------------------------------------------------------------
    tf_expand = ExpandTransform(max_ratio=4.0, mean_value=[104, 117, 123])
    tf_rnd_expand = RandomTransform(prob=expand_prob, transform=tf_expand)

    #---------------------------------------------------------------------------
    # Samplers
    #---------------------------------------------------------------------------
    samplers = [
        SamplerTransform(sample=False),
        build_sampler(0.1, sampler_trials),
        build_sampler(0.3, sampler_trials),
        build_sampler(0.5, sampler_trials),
        build_sampler(0.7, sampler_trials),
        build_sampler(0.9, sampler_trials),
        build_sampler(1.0, sampler_trials)
    ]
    tf_sample_picker = SamplePickerTransform(samplers=samplers)

    #---------------------------------------------------------------------------
    # Horizontal flip
    #---------------------------------------------------------------------------
    tf_flip = HorizontalFlipTransform()
    tf_rnd_flip = RandomTransform(prob=0.5, transform=tf_flip)

    #---------------------------------------------------------------------------
    # Transform list
    #---------------------------------------------------------------------------
    transforms = [
        ImageLoaderTransform(),
        #tf_rnd_brightness,
        #tf_distort,
        #tf_rnd_reorder_channels,
        #tf_rnd_expand,
        #tf_sample_picker,
        #tf_rnd_flip,
        LabelCreatorTransform(preset=preset, num_classes=num_classes),
        tf_resize
    ]
    return transforms

#-------------------------------------------------------------------------------
def build_valid_transforms(preset, num_classes):
    tf_resize = ResizeTransform(width=preset.image_size.w,
                                height=preset.image_size.h,
                                algorithms=[cv2.INTER_LINEAR])
    transforms = [
        ImageLoaderTransform(),
        LabelCreatorTransform(preset=preset, num_classes=num_classes),
        tf_resize
    ]
    return transforms

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Process a dataset for SSD')
    parser.add_argument('--data-source', default='pascal_voc2',
                        help='data source')                                       # LOAD TEXT DATASET
    parser.add_argument('--data-dir', default='E:\\#gp\\trainKOLO',
                        help='data directory')
    parser.add_argument('--validation-fraction', type=float, default=0.025,
                        help='fraction of the data to be used for validation')
    parser.add_argument('--expand-probability', type=float, default=0.5,
                        help='probability of running sample expander')
    parser.add_argument('--sampler-trials', type=int, default=10,
                        help='number of time a sampler tries to find a sample')
    parser.add_argument('--annotate', type=str2bool, default='False',
                        help="Annotate the data samples")
    parser.add_argument('--compute-td', type=str2bool, default='True',
                        help="Compute training data")
    parser.add_argument('--preset', default='vgg300',
                        choices=['vgg300', 'vgg512'],
                        help="The neural network preset")
    parser.add_argument('--process-test', type=str2bool, default='True',
                        help="process the test dataset")
    args = parser.parse_args()

    print('[i] Data source:          ', args.data_source)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] Validation fraction:  ', args.validation_fraction)
    print('[i] Expand probability:   ', args.expand_probability)
    print('[i] Sampler trials:       ', args.sampler_trials)
    print('[i] Annotate:             ', args.annotate)
    print('[i] Compute training data:', args.compute_td)
    print('[i] Preset:               ', args.preset)
    print('[i] Process test dataset: ', args.process_test)

    #---------------------------------------------------------------------------
    # Load the data source
    #---------------------------------------------------------------------------
    print('[i] Configuring the data source...')
    try:
        source = load_data_source(args.data_source)
        source.load_trainval_data(args.validation_fraction)
        if args.process_test:
            source.load_test_data()
        print('[i] # training samples:   ', source.num_train)
        print('[i] # validation samples: ', source.num_valid)
        print('[i] # testing samples:    ', source.num_test)
        print('[i] # classes:            ', source.num_classes)
    except (ImportError, AttributeError, RuntimeError) as e:
        print('[!] Unable to load data source:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Annotate samples
    #---------------------------------------------------------------------------
    if args.annotate:
        print('[i] Annotating samples...')
        annotate(args.data_dir, source.train_samples, source.colors, 'train')
        annotate(args.data_dir, source.valid_samples, source.colors, 'valid')
        if args.process_test:
            annotate(args.data_dir, source.test_samples,  source.colors, 'test ')

    #---------------------------------------------------------------------------
    # Compute the training data
    #---------------------------------------------------------------------------
    if args.compute_td:
        preset = get_preset_by_name(args.preset)
        with open(args.data_dir+'/train-samples.pkl', 'wb') as f:
            pickle.dump(source.train_samples, f)
        with open(args.data_dir+'/valid-samples.pkl', 'wb') as f:
            pickle.dump(source.valid_samples, f)

        with open(args.data_dir+'/training-data.pkl', 'wb') as f:
            data = {
                'preset': preset,
                'num-classes': source.num_classes,
                'colors': source.colors,
                'lid2name': source.lid2name,
                'lname2id': source.lname2id,
                'train-transforms': build_train_transforms(preset,
                                       source.num_classes, args.sampler_trials,
                                       args.expand_probability ),
                'valid-transforms': build_valid_transforms(preset,
                                                           source.num_classes)
            }
            pickle.dump(data, f)

    return 0

#if __name__ == '__main__':
#    sys.exit(main())
main()