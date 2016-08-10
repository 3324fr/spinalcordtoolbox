#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions and algorithm for image classification and segmentation
# using supervised machine learning
# python msct_keras_results.py
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

rcParams['legend.fontsize'] = 10

fname = sys.argv[1:]
print fname
cls_stats = pickle.load(open(fname[0], "rb"))

def plot_result(x, y, x_legend, text):
    """Plot result as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification ' + text + ' as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel(text)
    plt.grid(True)
    plt.plot(x, y)

# Plot evolution
plt.figure()
plt.subplot(2, 2, 1)
accuracy, n_examples = map(list, zip(*cls_stats['accuracy_history']))
plot_result(n_examples, accuracy, "training examples (#)", 'accuracy')
ax = plt.gca()
ax.set_ylim((0, 1))

plt.subplot(2, 2, 2)
precision, n_examples = map(list, zip(*cls_stats['precision_history']))
plot_result(n_examples, precision, "training examples (#)", 'precision')
ax = plt.gca()
ax.set_ylim((0, 1))

plt.subplot(2, 2, 3)
recall, n_examples = map(list, zip(*cls_stats['recall_history']))
plot_result(n_examples, recall, "training examples (#)", 'recall')
ax = plt.gca()
ax.set_ylim((0, 1))

plt.subplot(2, 2, 4)
fscore, n_examples = map(list, zip(*cls_stats['fscore_history']))
plot_result(n_examples, fscore, "training examples (#)", 'fscore')
ax = plt.gca()
ax.set_ylim((0, 1))

plt.show()

