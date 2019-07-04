﻿import argparse
import math
import sys
import os

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from average_precision import APCalculator, APs2mAP
#from training_data import TrainingData
#from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps , jaccard_overlap
from ssdvgg import SSDVGG
#from utils import *
from tqdm import tqdm

from collections import defaultdict
from collections import namedtuple
import pickle
import random
import math
import cv2

from collections import namedtuple, defaultdict
from math import sqrt, log, exp

import queue as q
#from data_queue import DataQueue
from copy import copy
import argparse

Label   = namedtuple('Label',   ['name', 'color'])
Size    = namedtuple('Size',    ['w', 'h'])
Point   = namedtuple('Point',   ['x', 'y'])
Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize'])
Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size'])
Score   = namedtuple('Score',   ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good'])

IMG_SIZE = Size(1000, 1000)

############################################################################################################################################################
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
#Label   = namedtuple('Label',   ['name', 'color'])
#Size    = namedtuple('Size',    ['w', 'h'])
#Point   = namedtuple('Point',   ['x', 'y'])
#Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize'])
#Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size'])
#Score   = namedtuple('Score',   ['idx', 'score'])
#Overlap = namedtuple('Overlap', ['best', 'good'])

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
#------------------------------------------------------------------------  transforms ---------------------------------

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




#--------------------------------------------- training data and dataqueue -----------------------------------------
class DataQueue:
    #---------------------------------------------------------------------------
    def __init__(self, img_template, label_template, maxsize):
        #-----------------------------------------------------------------------
        # Figure out the data tupes, sizes and shapes of both arrays
        #-----------------------------------------------------------------------
        self.img_dtype = img_template.dtype
        self.img_shape = img_template.shape
        self.img_bc = len(img_template.tobytes())
        self.label_dtype = label_template.dtype
        self.label_shape = label_template.shape
        self.label_bc = len(label_template.tobytes())

        #-----------------------------------------------------------------------
        # Make an array pool and queue
        #-----------------------------------------------------------------------
        self.array_pool = []
        self.array_queue = mp.Queue(maxsize)
        for i in range(maxsize):
            img_buff = mp.Array('c', self.img_bc, lock=False)
            img_arr = np.frombuffer(img_buff, dtype=self.img_dtype)
            img_arr = img_arr.reshape(self.img_shape)

            label_buff = mp.Array('c', self.label_bc, lock=False)
            label_arr = np.frombuffer(label_buff, dtype=self.label_dtype)
            label_arr = label_arr.reshape(self.label_shape)

            self.array_pool.append((img_arr, label_arr))
            self.array_queue.put(i)

        self.queue = mp.Queue(maxsize)

    #---------------------------------------------------------------------------
    def put(self, img, label, boxes, *args, **kwargs):
        #-----------------------------------------------------------------------
        # Check whether the params are consistent with the data we can store
        #-----------------------------------------------------------------------
        def check_consistency(name, arr, dtype, shape, byte_count):
            if type(arr) is not np.ndarray:
                raise ValueError(name + ' needs to be a numpy array')
            if arr.dtype != dtype:
                raise ValueError('{}\'s elements need to be of type {} but is {}' \
                                 .format(name, str(dtype), str(arr.dtype)))
            if arr.shape != shape:
                raise ValueError('{}\'s shape needs to be {} but is {}' \
                                 .format(name, shape, arr.shape))
            if len(arr.tobytes()) != byte_count:
                raise ValueError('{}\'s byte count needs to be {} but is {}' \
                                 .format(name, byte_count, len(arr.data)))

        check_consistency('img', img, self.img_dtype, self.img_shape,
                          self.img_bc)
        check_consistency('label', label, self.label_dtype, self.label_shape,
                          self.label_bc)

        #-----------------------------------------------------------------------
        # If we can not get the slot within timeout we are actually full, not
        # empty
        #-----------------------------------------------------------------------
        try:
            arr_id = self.array_queue.get(*args, **kwargs)
        except q.Empty:
            raise q.Full()

        #-----------------------------------------------------------------------
        # Copy the arrays into the shared pool
        #-----------------------------------------------------------------------
        self.array_pool[arr_id][0][:] = img
        self.array_pool[arr_id][1][:] = label
        self.queue.put((arr_id, boxes), *args, **kwargs)

    #---------------------------------------------------------------------------
    def get(self, *args, **kwargs):
        item = self.queue.get(*args, **kwargs)
        arr_id = item[0]
        boxes = item[1]

        img = np.copy(self.array_pool[arr_id][0])
        label = np.copy(self.array_pool[arr_id][1])

        self.array_queue.put(arr_id)

        return img, label, boxes

    #---------------------------------------------------------------------------
    def empty(self):
        return self.queue.empty()

#-------------------------------------------------------------------------------
class TrainingData:
    #---------------------------------------------------------------------------
    def __init__(self, data_dir):
        #-----------------------------------------------------------------------
        # Read the dataset info
        #-----------------------------------------------------------------------
        try:
            with open(data_dir+'/training-data.pkl', 'rb') as f:
                data = pickle.load(f)
            with open(data_dir+'/train-samples.pkl', 'rb') as f:
                train_samples = pickle.load(f)
            with open(data_dir+'/valid-samples.pkl', 'rb') as f:
                valid_samples = pickle.load(f)
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(str(e))

        nones = [None] * len(train_samples)
        train_samples = list(zip(nones, nones, train_samples))
        nones = [None] * len(valid_samples)
        valid_samples = list(zip(nones, nones, valid_samples))

        #-----------------------------------------------------------------------
        # Set the attributes up
        #-----------------------------------------------------------------------
        self.preset          = data['preset']
        self.num_classes     = data['num-classes']
        self.label_colors    = data['colors']
        self.lid2name        = data['lid2name']
        self.lname2id        = data['lname2id']
        self.train_tfs       = data['train-transforms']
        self.valid_tfs       = data['valid-transforms']
        self.train_generator = self.__batch_generator(train_samples,self.train_tfs)
        self.valid_generator = self.__batch_generator(valid_samples,self.valid_tfs)
        self.num_train       = len(train_samples)
        self.num_valid       = len(valid_samples)
        self.train_samples   = list(map(lambda x: x[2], train_samples))
        self.valid_samples   = list(map(lambda x: x[2], valid_samples))
        
    #---------------------------------------------------------------------------
    def __batch_generator(self, sample_list_, transforms):
        image_size = (self.preset.image_size.w, self.preset.image_size.h)
       
        #images, labels, gt_boxes = process_samples
        #-----------------------------------------------------------------------
        def run_transforms(sample):
            args = sample
            for t in transforms:
                args = t(*args)
            return args

        #-----------------------------------------------------------------------

        #########READ DATASET FROM FOLDER IMAGES & LABELS & BOUNDINGBOXES
        def process_samples(samples):
            images = []
            labels = []
            gt_boxes = []
            for s in samples:
                done = False
                counter = 0
                while not done and counter < 50:
                    image, label, gt = run_transforms(s)
                    num_bg = np.count_nonzero(label[:, self.num_classes])
                    done = num_bg < label.shape[0]
                    counter += 1
                
                images.append(image.astype(np.float32))
                labels.append(label.astype(np.float32))
                gt_boxes.append(gt.boxes)

            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            return images, labels, gt_boxes

        #-----------------------------------------------------------------------
        def batch_producer(sample_queue, batch_queue):
            while True:
                #---------------------------------------------------------------
                # Process the sample
                #---------------------------------------------------------------
                try:
                    samples = sample_queue.get(timeout=1)
                except q.Empty:
                    break

                images, labels, gt_boxes = process_samples(samples)

                #---------------------------------------------------------------
                # Pad the result in the case where we don't have enough samples
                # to fill the entire batch
                #---------------------------------------------------------------
                if images.shape[0] < batch_queue.img_shape[0]:
                    images_norm = np.zeros(batch_queue.img_shape,
                                           dtype=np.float32)
                    labels_norm = np.zeros(batch_queue.label_shape,
                                           dtype=np.float32)
                    images_norm[:images.shape[0]] = images
                    labels_norm[:images.shape[0]] = labels
                    batch_queue.put(images_norm, labels_norm, gt_boxes)
                else:
                    batch_queue.put(images, labels, gt_boxes)

        #-----------------------------------------------------------------------
        def gen_batch(batch_size, num_workers=0):
            sample_list = copy(sample_list_)
            random.shuffle(sample_list)

            #-------------------------------------------------------------------
            # Set up the parallel generator
            #-------------------------------------------------------------------
            if num_workers > 0:
                #---------------------------------------------------------------
                # Set up the queues
                #---------------------------------------------------------------
                img_template = np.zeros((batch_size, self.preset.image_size.h,
                                         self.preset.image_size.w, 3),
                                        dtype=np.float32)
                label_template = np.zeros((batch_size, self.preset.num_anchors,
                                           self.num_classes+5),
                                          dtype=np.float32)
                max_size = num_workers*5
                n_batches = int(math.ceil(len(sample_list_)/batch_size))
                sample_queue = mp.Queue(n_batches)
                batch_queue = DataQueue(img_template, label_template, max_size)

                #---------------------------------------------------------------
                # Set up the workers. Make sure we can fork safely even if
                # OpenCV has been compiled with CUDA and multi-threading
                # support.
                #---------------------------------------------------------------
                workers = []
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                cv2_num_threads = cv2.getNumThreads()
                cv2.setNumThreads(1)
                for i in range(num_workers):
                    args = (sample_queue, batch_queue)

                    w = mp.Process(target=batch_producer, args=args)
                    workers.append(w)
                    w.start()
                del os.environ['CUDA_VISIBLE_DEVICES']
                cv2.setNumThreads(cv2_num_threads)

                #---------------------------------------------------------------
                # Fill the sample queue with data
                #---------------------------------------------------------------
                for offset in range(0, len(sample_list), batch_size):
                    samples = sample_list[offset:offset+batch_size]
                    sample_queue.put(samples)

                #---------------------------------------------------------------
                # Return the data
                #---------------------------------------------------------------
                for offset in range(0, len(sample_list), batch_size):
                    images, labels, gt_boxes = batch_queue.get()
                    num_items = len(gt_boxes)
                    yield images[:num_items], labels[:num_items], gt_boxes

                #---------------------------------------------------------------
                # Join the workers
                #---------------------------------------------------------------
                for w in workers:
                    w.join()

            #-------------------------------------------------------------------
            # Return a serial generator
            #-------------------------------------------------------------------
            else:
                for offset in range(0, len(sample_list), batch_size):
                    samples = sample_list[offset:offset+batch_size]
                    images, labels, gt_boxes = process_samples(samples)
                    yield images, labels, gt_boxes

        sample_list = copy(sample_list_)
        for offset in range(0, len(sample_list), 40):
                    samples = sample_list[offset:offset+40]
                    images, labels, gt_boxes = process_samples(samples)
                    #yield images, labels, gt_boxes 
        return gen_batch



#-------------------------------------------------------------------------------
def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Train the SSD')
    parser.add_argument('--name', default='E:\\#gp\\ramadan3',
                        help='project name')
    parser.add_argument('--data-dir', default='E:\\#gp\\trainKOLO',
                        help='data directory')
    parser.add_argument('--vgg-dir', default='vgg_graph',
                        help='directory for the VGG-16 model')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--tensorboard-dir', default="tb2",
                        help='name of the tensorboard data directory')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='checkpoint interval')
    parser.add_argument('--lr-values', type=str, default='0.00075;0.0001;0.00001',
                        help='learning rate values')
    parser.add_argument('--lr-boundaries', type=str, default='320000;400000',
                        help='learning rate chage boundaries (in batches)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='L2 normalization factor')
    parser.add_argument('--continue-training', type=str2bool, default='false',
                        help='continue training from the latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of parallel generators')

    args = parser.parse_args()

    print('[i] Project name:         ', args.name)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] VGG directory:        ', args.vgg_dir)
    print('[i] # epochs:             ', args.epochs)
    print('[i] Batch size:           ', args.batch_size)
    print('[i] Tensorboard directory:', args.tensorboard_dir)
    print('[i] Checkpoint interval:  ', args.checkpoint_interval)
    print('[i] Learning rate values: ', args.lr_values)
    print('[i] Learning rate boundaries: ', args.lr_boundaries)
    print('[i] Momentum:             ', args.momentum)
    print('[i] Weight decay:         ', args.weight_decay)
    print('[i] Continue:             ', args.continue_training)
    print('[i] Number of workers:    ', args.num_workers)

    #---------------------------------------------------------------------------
    # Find an existing checkpoint
    #---------------------------------------------------------------------------
    start_epoch = 0
    if args.continue_training:
        state = tf.train.get_checkpoint_state(args.name)
        if state is None:
            print('[!] No network state found in ' + args.name)
            return 1

        ckpt_paths = state.all_model_checkpoint_paths
        if not ckpt_paths:
            print('[!] No network state found in ' + args.name)
            return 1

        last_epoch = None
        checkpoint_file = None
        for ckpt in ckpt_paths:
            ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = int(ckpt_num)
            except ValueError:
                continue
            if last_epoch is None or last_epoch < ckpt_num:
                last_epoch = ckpt_num
                checkpoint_file = ckpt

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph', metagraph_file)
            return 1
        start_epoch = last_epoch

    #---------------------------------------------------------------------------
    # Create a project directory
    #---------------------------------------------------------------------------
    #else:
        #try:
            #print('[i] Creating directory {}...'.format(args.name))
            #os.makedirs(args.name)
        #except (IOError) as e:
        #    print('[!]', str(e))
        #    return 1

    print('[i] Starting at epoch:    ', start_epoch+1)

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the training data...')
    try:
        td = TrainingData(args.data_dir)
        print('[i] # training samples:   ', td.num_train)
        print('[i] # validation samples: ', td.num_valid)
        print('[i] # classes:            ', td.num_classes)
        print('[i] Image size:           ', td.preset.image_size)
       
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data:', str(e))
       
        return 1

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    with tf.Session() as sess:
        print('[i] Creating the model...')
        n_train_batches = int(math.ceil(td.num_train/args.batch_size))
        n_valid_batches = int(math.ceil(td.num_valid/args.batch_size))

        global_step = None
        if start_epoch == 0:
            lr_values = args.lr_values.split(';')
            try:
                lr_values = [float(x) for x in lr_values]
            except ValueError:
                print('[!] Learning rate values must be floats')
                sys.exit(1)

            lr_boundaries = args.lr_boundaries.split(';')
            try:
                lr_boundaries = [int(x) for x in lr_boundaries]
            except ValueError:
                print('[!] Learning rate boundaries must be ints')
                sys.exit(1)

            ret = compute_lr(lr_values, lr_boundaries)
            learning_rate, global_step = ret

        net = SSDVGG(sess, td.preset)
        if start_epoch != 0:
            net.build_from_metagraph(metagraph_file, checkpoint_file)
            net.build_optimizer_from_metagraph()
        else:
            net.build_from_vgg(args.vgg_dir, td.num_classes)
            net.build_optimizer(learning_rate=learning_rate,
                                global_step=global_step,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

        initialize_uninitialized_variables(sess)

        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        summary_writer = tf.summary.FileWriter(args.tensorboard_dir,
                                               sess.graph)



        saver = tf.train.Saver(max_to_keep=20)

        anchors = get_anchors_for_preset(td.preset)
        training_ap_calc = APCalculator()
        validation_ap_calc = APCalculator()

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        training_ap = PrecisionSummary(sess, summary_writer, 'training',
                                       td.lname2id.keys() , restore)
        validation_ap = PrecisionSummary(sess, summary_writer, 'validation',
                                         td.lname2id.keys(), restore)

        training_imgs = ImageSummary(sess, summary_writer, 'training',
                                     td.label_colors, restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation',
                                       td.label_colors, restore)

        training_loss = LossSummary(sess, summary_writer, 'training',
                                    td.num_train, restore)
        validation_loss = LossSummary(sess, summary_writer, 'validation',
                                      td.num_valid, restore)

        #-----------------------------------------------------------------------
        # Get the initial snapshot of the network
        #-----------------------------------------------------------------------
        net_summary_ops = net.build_summaries(restore)
        if start_epoch == 0:
            net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(net_summary, 0)
        summary_writer.flush()

        #-----------------------------------------------------------------------
        # Cycle through the epoch
        #-----------------------------------------------------------------------
        print('[i] Training...')
        for e in range(start_epoch, args.epochs):
            training_imgs_samples = []
            validation_imgs_samples = []

            #-------------------------------------------------------------------
            # Train
            #-------------------------------------------------------------------
            generator = td.train_generator(args.batch_size)
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for x, y, gt_boxes in tqdm(generator, total=n_train_batches,
                                       desc=description, unit='batches'):

                if len(training_imgs_samples) < 3:
                    saved_images = np.copy(x[:3])

                feed = {net.image_input: x,
                        net.labels: y}
                result, loss_batch, _ = sess.run([net.result, net.losses,
                                                  net.optimizer],
                                                 feed_dict=feed)

                if math.isnan(loss_batch['confidence']):
                    print('[!] Confidence loss is NaN.')

                training_loss.add(loss_batch, x.shape[0])

                if e == 0: continue

                for i in range(result.shape[0]):
                    boxes = decode_boxes(result[i], anchors, 0.5, td.lid2name)
                    boxes = suppress_overlaps(boxes)
                    training_ap_calc.add_detections(gt_boxes[i], boxes)

                    if len(training_imgs_samples) < 3:
                        training_imgs_samples.append((saved_images[i], boxes))

            #-------------------------------------------------------------------
            # Validate
            #-------------------------------------------------------------------
            generator = td.valid_generator(args.batch_size)
            description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)

            for x, y, gt_boxes in tqdm(generator, total=n_valid_batches,
                                       desc=description, unit='batches'):
                feed = {net.image_input: x,
                        net.labels: y}
                result, loss_batch = sess.run([net.result, net.losses],
                                              feed_dict=feed)

                validation_loss.add(loss_batch,  x.shape[0])

                if e == 0: continue

                for i in range(result.shape[0]):
                    boxes = decode_boxes(result[i], anchors, 0.5, td.lid2name)
                    boxes = suppress_overlaps(boxes)
                    validation_ap_calc.add_detections(gt_boxes[i], boxes)

                    if len(validation_imgs_samples) < 3:
                        validation_imgs_samples.append((np.copy(x[i]), boxes))

            #-------------------------------------------------------------------
            # Write summaries
            #-------------------------------------------------------------------
            training_loss.push(e+1)
            validation_loss.push(e+1)

            net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(net_summary, e+1)

            APs = training_ap_calc.compute_aps()
            mAP = APs2mAP(APs)
            training_ap.push(e+1, mAP, APs)

            APs = validation_ap_calc.compute_aps()
            mAP = APs2mAP(APs)
            validation_ap.push(e+1, mAP, APs)

            training_ap_calc.clear()
            validation_ap_calc.clear()

            training_imgs.push(e+1, training_imgs_samples)
            validation_imgs.push(e+1, validation_imgs_samples)

            summary_writer.flush()

            #-------------------------------------------------------------------
            # Save a checktpoint
            #-------------------------------------------------------------------
            if (e+1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
                saver.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)

        checkpoint = '{}/final.ckpt'.format(args.name)
        saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:', checkpoint)

    return 0
#if __name__ == '__main__':
#    sys.exit(main())
main()
#################################################################################################################################################
