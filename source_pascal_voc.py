import random
import math
import cv2
import os
#import DataReading as SVTDS
from collections import namedtuple, defaultdict
from math import sqrt, log, exp

import argparse

import errno
import numpy as np


#from utils import Label, Box, Sample, Size
#from utils import rgb2bgr, abs2prop

from glob import glob
from tqdm import tqdm
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


#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
label_defs = [Label('0',   rgb2bgr((111,     74,   0))),
    Label('1',   rgb2bgr((111,     74,   0))),
    Label('2',   rgb2bgr((111,     74,   0))),
    Label('3',   rgb2bgr((111,     74,   0))),
    Label('4',   rgb2bgr((111,     74,   0))),
    Label('5',   rgb2bgr((111,     74,   0))),
    Label('6',   rgb2bgr((111,     74,   0))),
    Label('7',   rgb2bgr((111,     74,   0))),
    Label('8',   rgb2bgr((111,     74,   0))),
    Label('9',   rgb2bgr((111,     74,   0))),
    Label('A',   rgb2bgr((111,     74,   0))),
    Label('B',   rgb2bgr((111,     74,   0))),
    Label('C',   rgb2bgr((111,     74,   0))),
    Label('D',   rgb2bgr((111,     74,   0))),
    Label('E',   rgb2bgr((111,     74,   0))),
    Label('F',   rgb2bgr((111,     74,   0))),
    Label('G', rgb2bgr((111, 74, 0))),
    Label('H', rgb2bgr((111, 74, 0))),
    Label('I', rgb2bgr((111, 74, 0))),
    Label('J', rgb2bgr((111, 74, 0))),
    Label('K', rgb2bgr((111, 74, 0))),
    Label('L', rgb2bgr((111, 74, 0))),
    Label('M', rgb2bgr((111, 74, 0))),
    Label('N', rgb2bgr((111, 74, 0))),
    Label('O', rgb2bgr((111, 74, 0))),
    Label('P', rgb2bgr((111, 74, 0))),
    Label('Q', rgb2bgr((111, 74, 0))),
    Label('R', rgb2bgr((111, 74, 0))),
    Label('S', rgb2bgr((111, 74, 0))),
    Label('T', rgb2bgr((111, 74, 0))),
    Label('U', rgb2bgr((111, 74, 0))),
    Label('V', rgb2bgr((111, 74, 0))),
    Label('W', rgb2bgr((111, 74, 0))),
    Label('X', rgb2bgr((111, 74, 0))),
    Label('Y', rgb2bgr((111, 74, 0))),
    Label('Z', rgb2bgr((111, 74, 0)))]



#-------------------------------------------------------------------------------
class PascalVOCSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.num_classes   = len(label_defs)
        self.colors        = {l.name: l.color for l in label_defs}
        self.lid2name      = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id      = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []

    #---------------------------------------------------------------------------
   
    def reading_charsTEST(self,Directory,ImageDirctory):
        samples = []
        Directory =os.listdir(Directory)
        imgdir=os.listdir(ImageDirctory)
        nput='E:/#gp/mickeyCH/testann/'

        for i, img in enumerate(imgdir):
            FileName = img.split('.')[0]
            FileName=FileName+ ".txt"
            ThisImage = cv2.imread(ImageDirctory + img)
            ImageSize = Size(ThisImage.shape[1], ThisImage.shape[0])
            boxes = []
            FileGt = open(nput + FileName, "r")
            FileLines = FileGt.readlines()
            for line in FileLines:
                label= line.split()[0]
                xmin = float((line.split())[1])
                ymin = float((line.split())[2])
                width = float((line.split())[3])
                height = float((line.split())[4])

                xmax = width + xmin
                ymax = ymin + height
                center, size = abs2prop(xmin, xmax, ymin, ymax, ImageSize)
                #lab=self.labels(label)
                box = Box(label, self.lname2id[label.upper()], center, size)
                boxes.append(box)
            sample = Sample(ImageDirctory + img, boxes, ImageSize)
            samples.append(sample)

        print(samples[0])
        return samples

    #------------------------------------------------------------------ train & val -----------------------------------------

    def reading_chars(self,Directory,ImageDirctory):
        samples = []
        Directory =os.listdir(Directory)
        imgdir=os.listdir(ImageDirctory)
        nput='E:/#gp/annotation/'

        for i, img in enumerate(imgdir):
            FileName = img.split('.')[0]
            FileName=FileName+ ".txt"
            ThisImage = cv2.imread(ImageDirctory + img)
            ImageSize = Size(ThisImage.shape[1], ThisImage.shape[0])
            boxes = []
            FileGt = open(nput + FileName, "r")
            FileLines = FileGt.readlines()
            for line in FileLines:
                label= line.split()[0]
                xmin = float((line.split())[1])
                ymin = float((line.split())[2])
                width = float((line.split())[3])
                height = float((line.split())[4])

                xmax = width + xmin
                ymax = ymin + height
                center, size = abs2prop(xmin, xmax, ymin, ymax, ImageSize)
                #lab=self.labels(label)
                box = Box(label, self.lname2id[label.upper()], center, size)
                boxes.append(box)
            sample = Sample(ImageDirctory + img, boxes, ImageSize)
            samples.append(sample)
       # print(samples[0])
        return samples
    #---------------------------------------------------------------------------
    def load_trainval_data(self, valid_fraction):
       

        #-----------------------------------------------------------------------
        # Process the samples defined in the relevant file lists
        #-----------------------------------------------------------------------
        train_annot = []
        train_samples = []
        root='E:/#gp/trainimages/'
        #name = 'trainval_'+vocid
        root2='E:/#gp/annotation/'
        train_samples = self.reading_chars(root2,root)

        root = 'E:/#gp/validimages/'
    
        root2='E:/#gp/annotation/'
        
        valid_samples =  self.reading_chars(root2,root)

        #-----------------------------------------------------------------------
        # Final set up and sanity check
        #-----------------------------------------------------------------------
        self.valid_samples = valid_samples
        self.train_samples = train_samples

        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' )

        if valid_fraction > 0:
            if len(self.valid_samples) == 0:
                raise RuntimeError('No validation samples found in ' )

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)

    #---------------------------------------------------------------------------




    
    

    def load_test_data(self):

        root = 'E:/#gp/mickeyCH/test2/'
        root2='E:/#gp/mickeyCH/testann/'
        #annot = self.__build_annotation_list(root2)
        self.test_samples  = self.reading_charsTEST(root2,root)
        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in ' )

        self.num_test  = len(self.test_samples)

#-------------------------------------------------------------------------------
def get_source():
    return PascalVOCSource()


####3shan nrano
#PascalVOCSource.load_trainval_data(get_source(),0)


