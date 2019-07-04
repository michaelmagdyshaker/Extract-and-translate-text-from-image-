import argparse
import pickle
import math
import sys
import cv2
import os
from math import *
from collections import defaultdict ,namedtuple
import tensorflow as tf
import numpy as np
from average_precision import APCalculator, APs2mAP
#from pascal_summary import PascalSummary
#from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps ,draw_box2 , draw_box 
from ssdvgg import SSDVGG
#from utils import * #str2bool, load_data_source, draw_box , prop2abs 
from tqdm import tqdm
from operator import itemgetter
import random
from tkinter import *
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image
from tkinter import filedialog


from googletrans import Translator
import pyarabic.araby as araby
from module1 import *
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import argparse
import arabic_reshaper
import PIL
# install: pip install python-bidi
from bidi.algorithm import get_display
from collections import Counter


if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)

#-------------------------------------------------------------------------------
def sample_generator(samples, image_size, batch_size):
    image_size = (image_size.w, image_size.h)
    for offset in range(0, len(samples), batch_size): 
        files = samples[offset:offset+batch_size]
        images = []
        idxs   = []
        for i, image_file in enumerate(files):
            image = cv2.resize(cv2.imread(image_file), image_size)         #################"E:/#gp/mickeyCH/test/"
            images.append(image.astype(np.float32))
            idxs.append(offset+i)
        yield np.array(images), idxs

#------------------------------------------------- ssd utlis -----------------------------------------------------
def initialize_uninitialized_variables(sess):
 
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

    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def rgb2bgr(tpl):

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

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-------------------------------------------------------------------------------
def abs2prop(xmin, xmax, ymin, ymax, imgsize):

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

    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def rgb2bgr(tpl):

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
 
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-------------------------------------------------------------------------------
def abs2prop(xmin, xmax, ymin, ymax, imgsize):
  
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


#---------------------------------------------- summary ----------------------------------------------------

Detection = namedtuple('Detection', ['fileid', 'confidence', 'left', 'top',
                                     'right', 'bottom'])

#-------------------------------------------------------------------------------
class PascalSummary:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.boxes = defaultdict(list)

    #---------------------------------------------------------------------------
    def add_detections(self, filename, boxes):
        fileid = os.path.basename(filename)
        fileid = ''.join(fileid.split('.')[:-1])
        img = cv2.imread(filename)
        img_size = Size(img.shape[1], img.shape[0])
        for conf, box in boxes:
            xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
            if xmin < 0: xmin = 0
            if xmin >= img_size.w: xmin = img_size.w-1
            if xmax < 0: xmax = 0
            if xmax >= img_size.w: xmax = img_size.w-1
            if ymin < 0: ymin = 0
            if ymin >= img_size.h: ymin = img_size.h-1
            if ymax < 0: ymax = 0
            if ymax >= img_size.h: ymax = img_size.h-1
            det = Detection(fileid, conf, float(xmin+1), float(ymin+1), float(xmax+1), float(ymax+1))
            self.boxes[box.label].append(det)

    #---------------------------------------------------------------------------
    def write_summary(self, target_dir):
        for k, v in self.boxes.items():
            filename = target_dir+'/comp4_det_test_'+k+'.txt'
            with open(filename, 'w') as f:
                for det in v:
                    line = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n" \
                      .format(det.fileid, det.confidence, det.left, det.top,
                              det.right, det.bottom)
                    f.write(line)


#-------------------------------  collect the text from label of characters model --------------------------
def word (listaa) :
   if listaa != []:
        strWord = listaa[0][0]
        deff =0
        sum =0
        horizntal = False
        co = 0
        for i in range(1 , len(listaa)):
            if (listaa[i][1]== listaa [i-1][1] ):
                co+=1

        if (co == len(listaa)-1):
            horizntal = True 

        if (horizntal == False):
            print("-------------   sorted   ----------------------")
            listaa=sorted(listaa,key=itemgetter(1))
           # print(listaa)
            strWord = listaa[0][0]
           
            for i in range(1 , len(listaa)):
                deff1 =listaa[i][1]- listaa [i-1][2] 
                if deff1<0:
                    deff1=0
                sum+= deff1

            avg = sum / 3

            for i in range (1 , len(listaa)) :
                deff =listaa[i][1]- listaa [i-1][2] 
                if (deff > avg and (deff-avg) >200 ) :
                    strWord += " "
                    strWord += listaa[i][0]
                else :
                    strWord += listaa[i][0]

        else :
              listaa=sorted(listaa,key=itemgetter(3))
              # print(listaa)
              strWord = listaa[0][0]
              for i in range(1 , len(listaa)):
                deff1 =listaa[i][3]- listaa [i-1][4] 
                if deff1<0:
                    deff1=0
                sum+= deff1

              avg = sum / len(listaa)

              for i in range (1 , len(listaa)) :
                deff =listaa[i][3]- listaa [i-1][4] 
                if (deff > avg and (deff-avg) >6) :
                    strWord += " "
                    strWord += listaa[i][0]
                else :
                    strWord += listaa[i][0]
   else:
       strWord="NULL"

   return strWord


#--------------------------------- BACHGROUNG DETECTION ---------------------------------------
class BackgroundColorDetector():
    def __init__(self, imageLoc):
        self.img = cv2.imread(imageLoc, 1)
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w*self.h
 
    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                RGB = (self.img[x,y,2],self.img[x,y,1],self.img[x,y,0])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1
 
    def average_colour(self):
        red = 0; green = 0; blue = 0;
        sample = 10
        for top in range(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]
 
        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
        print ("Average RGB for top ten is: (", average_red, ", ", average_green, ", ", average_blue, ")")
        return  average_red, average_green, average_blue
 
    def twenty_most_common(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)
        for rgb, value in self.number_counter:
            print (rgb, value, ((float(value)/self.total_pixels)*100))
 
    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = (float(self.number_counter[0][1])/self.total_pixels)
        print (self.percentage_of_first)
        #if self.percentage_of_first > 0.5:
        return self.number_counter[0][0]
        #else:
        #    average_red, average_green, average_blue= self.average_colour()
        #    RGB=(int(average_red), int(average_green), int(average_blue))
        #    return RGB




def most_frequent_colour(img):
    #img = np.array(img)
    width, height = img.shape[:2]

    r_total = 0
    g_total = 0
    b_total = 0

    count = 0
   # img=PIL.Image.fromarray(img)
    for x in range(0, width):
        for y in range(0, height):
            r, g, b = img[x,y]
            r_total += r
            g_total += g
            b_total += b
            count += 1
    rgb=(int(r_total/count), int(g_total/count), int(b_total/count))
    return rgb

def mainTEXT():
    #---------------------------------------------------------------------------
    # Parse commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='SSD inference')
    parser.add_argument("files", nargs="*")
    parser.add_argument('--name', default='E:\\#gp\\ramadan',
                        help='project name')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to restore; -1 is the most recent')
    parser.add_argument('--training-data',
                        default='E:\\#gp\\trainn\\training-data.pkl',
                        help='Information about parameters used for training')
    parser.add_argument('--output-dir', default='test-outFinal',
                        help='directory for the resulting images')
    parser.add_argument('--annotate', type=str2bool, default='True',
                        help="Annotate the data samples")
    parser.add_argument('--dump-predictions', type=str2bool, default='True',
                        help="Dump raw predictions")
    parser.add_argument('--compute-stats', type=str2bool, default='True',
                        help="Compute the mAP stats")
    parser.add_argument('--data-source', default='pascal_voc2',
                        help='Use test files from the data source')
    parser.add_argument('--data-dir', default='E:\\#gp\\trainn',
                        help='Use test files from the data source')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--sample', default='test', 

                        choices=['test', 'trainval'], help='sample to run on')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--pascal-summary', type=str2bool, default='False',
                        help='dump the detections in Pascal VOC format')

    args = parser.parse_args()

    #---------------------------------------------------------------------------
    # Print parameters
    #---------------------------------------------------------------------------
    print('[i] Project name:      ', args.name)
    print('[i] Training data:     ', args.training_data)
    print('[i] Batch size:        ', args.batch_size)
    print('[i] Data source:       ', args.data_source)
    print('[i] Data directory:    ', args.data_dir)
    print('[i] Output directory:  ', args.output_dir)
    print('[i] Annotate:          ', args.annotate)
    print('[i] Dump predictions:  ', args.dump_predictions)
    print('[i] Sample:            ', args.sample)
    print('[i] Threshold:         ', args.threshold)
    print('[i] Pascal summary:    ', args.pascal_summary)

    #---------------------------------------------------------------------------
    # Check if we can get the checkpoint
    #---------------------------------------------------------------------------
#########################################################################3 
    #state = tf.train.get_checkpoint_state(args.name)

    #if state is None:
    #    print('[!] No network state found in ' + args.name)
    #    return 1

    #try:
    #   checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
    #except IndexError:
    #    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    #    return 1
    
    c=0
    #metagraph_file = checkpoint_file + '.meta'
    checkpoint_file='E:\\#gp\\ramadan\\final.ckpt'
    metagraph_file='E:\\#gp\\ramadan\\final.ckpt'+'.meta'
    if not os.path.exists(metagraph_file):
        print('[!] Cannot find metagraph ' + metagraph_file)
        return 1

    ##---------------------------------------------------------------------------
    ## Load the training data
    ##---------------------------------------------------------------------------
    try:
        with open(args.training_data, 'rb') as f:
            data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        num_classes = data['num-classes']
        image_size = preset.image_size
        anchors = get_anchors_for_preset(preset)
    except (FileNotFoundError, IOError, KeyError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Load the data source if defined
    #---------------------------------------------------------------------------
    compute_stats = False

    if args.data_source:
        print('[i] Configuring the data source...')
        try:
            source = load_data_source(args.data_source)
            if args.sample == 'test':
                source.load_test_data()
                num_samples = source.num_test
                samples     = source.test_samples
            else:
                source.load_trainval_data(args.data_dir, 0)
                num_samples = source.num_train
                samples = source.train_samples
            print('[i] # samples:         ', num_samples)
            print('[i] # classes:         ', source.num_classes)
        except (ImportError, AttributeError, RuntimeError) as e:
            print('[!] Unable to load data source:', str(e))
            return 1

        if args.compute_stats:
            compute_stats = True

    #---------------------------------------------------------------------------
    # Create a list of files to analyse and make sure that the output directory
    # exists
    #---------------------------------------------------------------------------
    files = []

    if source:
        for sample in samples:
            files.append(sample.filename)

    if not source:
        if args.files:
            files = args.files

        if not files:
            print('[!] No files specified')
            return 1

    #files = list(filter(lambda x: os.path.exists(x), files))
    if files:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    #---------------------------------------------------------------------------
    # Print model and dataset stats
    #---------------------------------------------------------------------------
    print('[i] Compute stats:     ', compute_stats)
    print('[i] Network checkpoint:', checkpoint_file)
    print('[i] Metagraph file:    ', metagraph_file)
    print('[i] Image size:        ', image_size)
    print('[i] Number of files:   ', len(files))
    
    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    if compute_stats:
        ap_calc = APCalculator()

    if args.pascal_summary:
        pascal_summary = PascalSummary()

    with tf.Session() as sess:
        print('[i] Creating the model...')
        net = SSDVGG(sess, preset)
        net.build_from_metagraph(metagraph_file, checkpoint_file)

        #-----------------------------------------------------------------------
        # Process the images
        #-----------------------------------------------------------------------
        generator = sample_generator(files, image_size, args.batch_size)
        n_sample_batches = int(math.ceil(len(files)/args.batch_size))
        description = '[i] Processing samples'

        for x, idxs in tqdm(generator, total=n_sample_batches,
                      desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    net.keep_prob:    1}
            enc_boxes = sess.run(net.result, feed_dict=feed)
            ####################################################################################      henaaaaaaaaaaaaaaaaaaaa walaaa ?? 
            #-------------------------------------------------------------------
            # Process the predictions
            #-------------------------------------------------------------------
            listaa = []
           
            for i in range(enc_boxes.shape[0]):
                boxes = decode_boxes(enc_boxes[i], anchors, args.threshold,
                                     lid2name, None)
                boxes = suppress_overlaps(boxes)[:200]


                filename = files[idxs[i]]
                basename = os.path.basename(filename)

                #---------------------------------------------------------------
                # Annotate samples
                #---------------------------------------------------------------
                listaa = []
               
                if args.annotate:
                    img = cv2.imread(filename)
                    
                    for box in boxes:
                        img_size=draw_box(img, box[1], colors[box[1].label])
                        c+=1
                        char = box[1][0];
            
                        xmin, xmax, ymin, ymax = prop2abs(box[1].center, box[1].size, img_size)
               
                        listaa.append([ xmin , xmax , ymin ,ymax , img])
                        
                        
                        crop=img[int(abs(ymin-30)):int(ymax+30) , int(abs(xmin-30)):int(xmax+30 )]
                        #cv2.imshow("cropped", crop)
                        path = 'E:/#gp/mickeyCH/test2/'
                        cv2.imwrite(os.path.join(path ,str(c)+ '.jpg'),crop)
                        f = open("E:/#gp/mickeyCH/testann/" +str(c) + ".txt",'w')
                    fn = args.output_dir+'/'+basename
                    print(filename)

                    cv2.imwrite(fn, img)

                #---------------------------------------------------------------
                # Dump the predictions
                #---------------------------------------------------------------
                if args.dump_predictions: 
                    raw_fn = args.output_dir+'/'+basename+'.npy'
                    np.save(raw_fn, enc_boxes[i])

                #---------------------------------------------------------------
                # Add predictions to the stats calculator and to the Pascal
                # summary
                #---------------------------------------------------------------
                if compute_stats:
                    ap_calc.add_detections(samples[idxs[i]].boxes, boxes)

                if args.pascal_summary:
                    pascal_summary.add_detections(filename, boxes)
                    


    #---------------------------------------------------------------------------
    # Compute and print the stats
    #---------------------------------------------------------------------------
    if compute_stats:
        aps = ap_calc.compute_aps()
        for k, v in aps.items():
            print('[i] AP [{0}]: {1:.3f}'.format(k, v))

        print('[i] mAP: {0:.3f}'.format(APs2mAP(aps)))

    #---------------------------------------------------------------------------
    # Write the pascal summary files
    #---------------------------------------------------------------------------
    if args.pascal_summary:
        pascal_summary.write_summary(args.output_dir)

    print('[i] All done.')
    return listaa












#-------------------------------------------------------------------------------
def mainCH(lista):
    #---------------------------------------------------------------------------
    # Parse commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='SSD inference2')
    parser.add_argument("files", nargs="*")
    parser.add_argument('--name', default='E:\\#gp\\ramadangana',
                        help='project name')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to restore; -1 is the most recent')
    parser.add_argument('--training-data',
                        default='E:\\#gp\\trainKOLO\\training-data.pkl',
                        help='Information about parameters used for training')
    parser.add_argument('--output-dir', default='test-outCHfinal',
                        help='directory for the resulting images')
    parser.add_argument('--annotate', type=str2bool, default='True',
                        help="Annotate the data samples")
    parser.add_argument('--dump-predictions', type=str2bool, default='True',
                        help="Dump raw predictions")
    parser.add_argument('--compute-stats', type=str2bool, default='True',
                        help="Compute the mAP stats")
    parser.add_argument('--data-source', default='pascal_voc',
                        help='Use test files from the data source')
    parser.add_argument('--data-dir', default='E:\\#gp\\trainKOLO',
                        help='Use test files from the data source')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--sample', default='test',
                        choices=['test', 'trainval'], help='sample to run on')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--pascal-summary', type=str2bool, default='False',
                        help='dump the detections in Pascal VOC format')

    args = parser.parse_args()

    #---------------------------------------------------------------------------
    # Print parameters
    #---------------------------------------------------------------------------
    print('[i] Project name:      ', args.name)
    print('[i] Training data:     ', args.training_data)
    print('[i] Batch size:        ', args.batch_size)
    print('[i] Data source:       ', args.data_source)
    print('[i] Data directory:    ', args.data_dir)
    print('[i] Output directory:  ', args.output_dir)
    print('[i] Annotate:          ', args.annotate)
    print('[i] Dump predictions:  ', args.dump_predictions)
    print('[i] Sample:            ', args.sample)
    print('[i] Threshold:         ', args.threshold)
    print('[i] Pascal summary:    ', args.pascal_summary)

    #---------------------------------------------------------------------------
    # Check if we can get the checkpoint
    #---------------------------------------------------------------------------
#########################################################################3 
    #state = tf.train.get_checkpoint_state(args.name)

    #if state is None:
    #    print('[!] No network state found in ' + args.name)
    #    return 1

    #try:
    #   checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
    #except IndexError:
    #    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    #    return 1

    #metagraph_file = checkpoint_file + '.meta'
    checkpoint_file='E:\\#gp\\ramadangana\\final.ckpt'
    metagraph_file='E:\\#gp\\ramadangana\\final.ckpt'+'.meta'
    if not os.path.exists(metagraph_file):
        print('[!] Cannot find metagraph ' + metagraph_file)
        return 1

    ##---------------------------------------------------------------------------
    ## Load the training data
    ##---------------------------------------------------------------------------
    try:
        with open(args.training_data, 'rb') as f:
            data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        num_classes = data['num-classes']
        image_size = preset.image_size
        anchors = get_anchors_for_preset(preset)
    except (FileNotFoundError, IOError, KeyError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Load the data source if defined
    #---------------------------------------------------------------------------
    compute_stats = False

    if args.data_source:
        print('[i] Configuring the data source...')
        try:
            source = load_data_source(args.data_source)
            if args.sample == 'test':
                source.load_test_data()
                num_samples = source.num_test
                samples     = source.test_samples
            else:
                source.load_trainval_data(args.data_dir, 0)
                num_samples = source.num_train
                samples = source.train_samples
            print('[i] # samples:         ', num_samples)
            print('[i] # classes:         ', source.num_classes)
        except (ImportError, AttributeError, RuntimeError) as e:
            print('[!] Unable to load data source:', str(e))
            return 1

        if args.compute_stats:
            compute_stats = True

    #---------------------------------------------------------------------------
    # Create a list of files to analyse and make sure that the output directory
    # exists
    #---------------------------------------------------------------------------
    files = []

    if source:
        for sample in samples:
            files.append(sample.filename)

    if not source:
        if args.files:
            files = args.files

        if not files:
            print('[!] No files specified')
            return 1

    if files:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    #---------------------------------------------------------------------------
    # Print model and dataset stats
    #---------------------------------------------------------------------------
    print('[i] Compute stats:     ', compute_stats)
    print('[i] Network checkpoint:', checkpoint_file)
    print('[i] Metagraph file:    ', metagraph_file)
    print('[i] Image size:        ', image_size)
    print('[i] Number of files:   ', len(files))
    
    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    if compute_stats:
        ap_calc = APCalculator()

    if args.pascal_summary:
        pascal_summary = PascalSummary()

    
    tf.reset_default_graph()
        
    with tf.Session() as sess:
        print('[i] Creating the model...')
        net = SSDVGG(sess, preset)

        net.build_from_metagraph(metagraph_file, checkpoint_file)

        #-----------------------------------------------------------------------
        # Process the images
        #-----------------------------------------------------------------------
        generator = sample_generator(files, image_size, args.batch_size)
        n_sample_batches = int(math.ceil(len(files)/args.batch_size))
        description = '[i] Processing samples'
        
        for x, idxs in tqdm(generator, total=n_sample_batches,
                      desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    net.keep_prob:    1}
            enc_boxes = sess.run(net.result, feed_dict=feed)
            ####################################################################################      henaaaaaaaaaaaaaaaaaaaa walaaa ?? 
            #-------------------------------------------------------------------
            # Process the predictions
            #-------------------------------------------------------------------
            listaa = []
            coun=0
            imgOld = lista[coun][4]   # old image (big image)
            for i in range(enc_boxes.shape[0]):
                boxes = decode_boxes(enc_boxes[i], anchors, args.threshold,
                                     lid2name, None)
                boxes = suppress_overlaps(boxes)[:200]


                filename = files[idxs[i]]
                basename = os.path.basename(filename)

                #---------------------------------------------------------------
                # Annotate samples
                #---------------------------------------------------------------
                listaa = []
                if args.annotate:
                    img = cv2.imread(filename)
                   
                    for box in boxes:
                       
                        img_size=draw_box(img, box[1], colors[box[1].label])
                       
                        char = box[1][0];
                        #h = box[1][3][1] *100;
                        #w = box[1][3][0] *100;
                        #xmin = box[1][2][0];
                        #ymin = box[1][2][1];
                        
                        xmin, xmax, ymin, ymax = prop2abs(box[1].center, box[1].size, img_size)
               
                        listaa.append([char, xmin , xmax , ymin ,ymax])
                        
                    fn = args.output_dir+'/'+basename
                    print(listaa)
                    wordStr = word(listaa)
                    
                    print(wordStr)
                    f = open("test.txt",'a')
                    f.write(wordStr)
                    f.write(' \n')
                    f.close()
                   
                    wordTrans=trans(wordStr)   ### translate the word which collected
                    
                    #BackgroundColor = BackgroundColorDetector('E:/#gp/mickeyCH/test2/1.jpg')
                    #print(BackgroundColor.detect())
                    #RGB =BackgroundColor.detect()
                    
                    imgBlur=cv2.imread('E:/#gp/mickeyCH/test2/'+str(coun+1)+'.jpg')
                    imgBlur=cv2.blur(img,(120,120))
                    f=most_frequent_colour(imgBlur)
                    x=(255-f[0],255-f[1],255-f[2]) 
                    img2 = Image.new('RGB', ((lista[coun][1]-lista[coun][0])+30, (lista[coun][3]-lista[coun][2])+30), color =f )  # make new image in h and w of the word
                                    
                    fnt = ImageFont.truetype('/Library/Fonts/arial.ttf', 100)
                    d = ImageDraw.Draw(img2)
                    text =wordTrans
                    reshaped_text = arabic_reshaper.reshape(text)    # correct its shape
                    bidi_text = get_display(reshaped_text)           # correct its direction
                    #print(bidi_text)
###############################################################################################

                    
                    # get boundary of this text
                    #textsize = cv2.getTextSize(text, fnt, 100, 2)[0]

                    ## get coords based on boundary
                    #textX = (img.shape[1] - textsize[0]) / 2
                    #textY = (img.shape[0] + textsize[1]) / 2
                    #img2 = PIL.Image.fromarray(img2) 
                   
                    d.text((5, 20 ) ,bidi_text,font=fnt, fill=x)
                    img2.save('pil_text_font.png')                            
                    imgOld = PIL.Image.fromarray(imgOld)                    
                    imgOld.paste(img2, (lista[coun][0], lista[coun][2]))     # make new image : old image with image of word translated in x and y
                    imgOld = np.array(imgOld)
                    cv2.imwrite(fn, img)
                    cv2.imwrite(os.path.join("C:/Users/Marina Georgy/Documents/Visual Studio 2013/Projects/PythonApplication7/PythonApplication7/tb/","1.jpg"), imgOld)
                    coun+=1

                #---------------------------------------------------------------
                # Dump the predictions
                #---------------------------------------------------------------
                if args.dump_predictions: 
                    raw_fn = args.output_dir+'/'+basename+'.npy'
                    np.save(raw_fn, enc_boxes[i])

                #---------------------------------------------------------------
                # Add predictions to the stats calculator and to the Pascal
                # summary
                #---------------------------------------------------------------
                if compute_stats:
                    ap_calc.add_detections(samples[idxs[i]].boxes, boxes)

                if args.pascal_summary:
                    pascal_summary.add_detections(filename, boxes)
                    


    #---------------------------------------------------------------------------
    # Compute and print the stats
    #---------------------------------------------------------------------------
    if compute_stats:
        aps = ap_calc.compute_aps()
        for k, v in aps.items():
            print('[i] AP [{0}]: {1:.3f}'.format(k, v))

        print('[i] mAP: {0:.3f}'.format(APs2mAP(aps)))

    #---------------------------------------------------------------------------
    # Write the pascal summary files
    #---------------------------------------------------------------------------
    if args.pascal_summary:
        pascal_summary.write_summary(args.output_dir)

    print('[i] All done.')
    return 0

def main():
    this = sys.modules[__name__]
    for n in dir():
      if n[0]!='_': 
        delattr(this, n)
    mainTEXT()
    this = sys.modules[__name__]
    for n in dir():
        if n[0]!='_': delattr(this, n)
    mainCH()
    return 0
#if __name__ == '__main__':
#    sys.exit(main())

#translator = Translator()
#translator = Translator(service_urls=[
#        'translate.google.com',
#        'translate.google.co.kr'])
#lista=[]
#lista=mainTEXT()
#mainCH(lista)
#trans("dog")

##------------------------------------------ GUI ----------------------------------------------------------------------

top = Tk()
top.title("Single layer")
top.geometry("1000x1000")
top.configure(bg='white')
top.title("Extract & Translate Text From The Image")



def file_read(fname):
        content_array = []
        with open(fname) as f:
                #Content_list is the list that contains the read lines.     
                for line in f:
                        content_array.append(line)
                print(content_array)
        return content_array

def open_loadimage():
    filename = filedialog.askopenfilename(title='open')
    img = cv2.imread(filename)

    imgName =os.path.basename(os.path.normpath(filename))
    cv2.imwrite(os.path.join('E:\\#gp\\gp\\neocr_dataset\\neocr_dataset\\test\\' ,"1 (1)"+ '.jpg'),img)

    x = Image.open(filename)
    img = x.resize((450, 330), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)  #####################33
    # Displaying it
    imglabel = tk.Label(top, image=img)
    imglabel.image = img
    imglabel.grid(column=0, row=1)
    lista=[]
    lista=mainTEXT()
    mainCH(lista)

#def openfn():
#    filename = filedialog.askopenfilename(title='open')
#    img = cv2.imread(filename)
#    imgName =os.path.basename(os.path.normpath(filename))
#    cv2.imwrite(os.path.join('E:\\#gp\\gp\\neocr_dataset\\neocr_dataset\\test\\' ,"1 (1)"+ '.jpg'),img)
#    #f = open("E:/#gp/mickeyCH/testann/" + "500"+str(c) + ".txt",'w')
#    return filename


#def open_img():
#    x = openfn()
#    img = Image.open(x)
#    img = img.resize((400, 250), Image.ANTIALIAS)
#    img = ImageTk.PhotoImage(img)
#    panel = tk.Label(top, image=img)
    
#    panel.image = img
#    panel.pack()
#    panel.place(x=50, y=50)

#    lista=[]
#    lista=mainTEXT()
#    mainCH(lista)



def boundingbox():
    # Setting it up
    x2 = Image.open("C:/Users/Marina Georgy/Documents/Visual Studio 2013/Projects/PythonApplication7/PythonApplication7/test-outFinal/"+"1 (1).jpg")
    img2 = x2.resize((450, 330), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img2)  ################

    # Displaying it
    imglabel2 = tk.Label(top, image=img2)
    imglabel2.image = img2
    imglabel2.grid(column=1, row=1)
    ###########################################################################33endof2

def character():
    # Setting it up
    
    x3 = Image.open("C:/Users/Marina Georgy/Documents/Visual Studio 2013/Projects/PythonApplication7/PythonApplication7/test-outCHfinal/"+"1.jpg")
    img3 = x3.resize((450, 330), Image.ANTIALIAS)
    img3 = ImageTk.PhotoImage(img3)  ##########################3

    # Displaying it
    imglabel3 = tk.Label(top, image=img3)
    imglabel3.image = img3
    imglabel3.grid(column=0, row=5)
    #############################################################################33endof3

def result():
    # Setting it up
    x4 = Image.open("C:/Users/Marina Georgy/Documents/Visual Studio 2013/Projects/PythonApplication7/PythonApplication7/tb/"+"1.jpg")
    img4 = x4.resize((450, 330), Image.ANTIALIAS)
    img4 = ImageTk.PhotoImage(img4)  ####################3

    # Displaying it
    imglabel4 = tk.Label(top, image=img4)
    imglabel4.image = img4
    imglabel4.grid(column=1, row=5)

#def open_img2():
#    #mainCH()
#    x = "C:/Users/Marina Georgy/Documents/Visual Studio 2013/Projects/PythonApplication7/PythonApplication7/test-outCHfinal/"+"1.jpg"
#    img = Image.open(x)
#    img = img.resize((400, 250), Image.ANTIALIAS)
#    img = ImageTk.PhotoImage(img)
#    panel = tk.Label(top, image=img)
#    panel.image = img
#    panel.pack()
#    panel.place(x=600, y=50)
#    #--------------------------- translate to file
    

inputimage=tk.Label(top,text="Enter your Image",font=("Times New Roman",15),justify=LEFT)
inputimage.grid(column=0,row=0)


##########################################################################endof1
#Label
boundingboxtext=tk.Label(top,text="Bounding box Result of text detection",font=("Times New Roman",15),justify=LEFT)
boundingboxtext.grid(column=1,row=0)



#Label
char=tk.Label(top,text="Enter your Image",font=("Result of Character Recognition",15),justify=LEFT)
char.grid(column=0,row=4)


#Label
output=tk.Label(top,text="After Translation",font=("Times New Roman",15),justify=LEFT)
output.grid(column=1,row=4)




button=Button(top,text="Load image",command=open_loadimage,bg="white")
button.grid(column=0,row=2)
button=Button(top,text="Show Text Detection",command=boundingbox,bg="white")
button.grid(column=1,row=2)
button=Button(top,text="Show Character Recognition",command=character,bg="white")
button.grid(column=0,row=6)
button=Button(top,text="Get the translation",command=result,bg="white")
button.grid(column=1,row=6)

top.mainloop()

#btn = Button(top,text="Load The Image",font=("Arial Bold", 12),bg="White", fg="brown")
#btn.pack()
#btn.place(x=50,y=15)
#btn.config(command = open_img)

#btn = Button(top,text="Get The Translation ",font=("Arial Bold", 12),bg="White", fg="brown")
#btn.pack()
#btn.place(x=600,y=15)
#btn.config(command = open_img2)

#var=StringVar()
#l=tk.Label(top, textvariable=var ,font=("Arial Bold", 14))
#var.set("Input Image")
#l.pack()
#l.place(x=20, y=450)

#var=StringVar()
#l=tk.Label(top, textvariable=var ,font=("Arial Bold", 14))
#var.set("Image with Bounding BOx")
#l.pack()
#l.place(x=400, y=450)


#bt2=Button(top, text="Quit",font=("Arial Bold", 12,),bg='brown')
#bt2.pack()
#bt2.place(x=750, y = 600)
#bt2.config(command = top.destroy)

#top.mainloop()
#----------------------------------------------------------- end gui -------------------------------------------------










