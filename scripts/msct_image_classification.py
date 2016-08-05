#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions and algorithm for image classification and segmentation
# using supervised machine learning
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
from msct_image import Image
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import neighbors, linear_model, cross_validation, svm
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.learning_curve import learning_curve
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import compute_class_weight

def extract_data(path_data, verbose=1):
    """
    Extract the images into a (samples, feature) matrix.
    """
    from sys import stdout
    ignore_list = ['.DS_Store']
    if verbose == 1:
        sct.printv('Extracting '+ path_data)
    cr = '\r'

    data = []
    list_data = []
    #images_folder = os.listdir(path_data)
    for root, dirs, files in os.walk(path_data):
        for fname_im in files:
            if fname_im in ignore_list:
                continue
            if 'seg' in fname_im or 'gmseg' in fname_im:
                continue
            list_data.append(os.path.join(root, fname_im))

    for i, fname_im in enumerate(list_data):
        if verbose == 1:
            stdout.write(cr)
            stdout.write(str(i) + '/' + str(len(list_data)))
        im_data = Image(fname_im)
        data.append(np.expand_dims(im_data.data.flatten(), axis=0))

    data_result = np.concatenate(data, axis=0)
    if verbose == 1:
        stdout.write(cr)
        print 'Matrix shape (samples, feature):', data_result.shape
    return data_result.astype(np.float32)


def extract_slices_from_image(fname_im, fname_seg=None):
    im_data = Image(fname_im)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    if fname_seg:
        im_seg = Image(fname_seg)

    data_im = []
    data_seg = []
    for k in range(nz):
        data_im.append(im_data.data[:, :, k])
        if fname_seg:
            data_seg.append(im_seg.data[:, :, k])

    if fname_seg:
        return data_im, data_seg
    else:
        return data_im

def extract_patch_from_slice(slice_im, slice_seg=None, patch_size=32, max_patches_factor=1):
    if slice_seg is not None:
        data_to_patch = np.stack((slice_im, slice_seg), axis=2)
    else:
        data_to_patch = slice_im
    max_patches = int(data_to_patch.shape[0] * data_to_patch.shape[1] / max_patches_factor)
    rng = np.random.RandomState(0)
    return extract_patches_2d(data_to_patch, (patch_size, patch_size), max_patches=max_patches, random_state=rng)


def extract_list_file_from_path(path_data):
    from sys import stdout
    ignore_list = ['.DS_Store']
    sct.printv('Extracting ' + path_data)
    cr = '\r'

    data = []
    list_data = []
    # images_folder = os.listdir(path_data)
    for root, dirs, files in os.walk(path_data):
        for fname_im in files:
            if fname_im in ignore_list:
                continue
            if 'seg' in fname_im or 'gmseg' in fname_im:
                continue
            f_seg = None
            for fname_seg in files:
                if fname_im[:-7] in fname_seg:
                    f_seg = fname_seg
            list_data.append([root + '/' + fname_im, root + '/' + f_seg])

    return list_data


def create_patches_imseg(list_data, verbose=1):
    patch_size = 32
    max_patches_factor = 250

    list_patches, list_classes = [], []

    for i, fname in enumerate(list_data):
        if verbose == 1:
            stdout.write(cr)
            stdout.write(str(i) + '/' + str(len(list_data)))

        data_im, data_seg = extract_slices_from_image(list_data[i][0], list_data[i][1])
        number_of_slices = len(data_im)

        for k in range(number_of_slices):
            patches = extract_patch_from_slice(data_im[k], data_seg[k], patch_size, max_patches_factor)
            number_of_patches = patches.shape[0]
            #print k, number_of_slices, number_of_patches
            for j in range(number_of_patches):
                patch_im = patches[j, :, :, 0]
                patch_seg = patches[j, :, :, 1]

                list_patches.append(np.expand_dims(patch_im.flatten(), axis=0))
                if patch_seg[int(patch_size/2), int(patch_size/2)] == 1:
                    list_classes.append(1)
                else:
                    list_classes.append(0)

    list_patches = np.concatenate(list_patches, axis=0)
    list_classes = np.array(list_classes)

    return list_patches, list_classes

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def stream_images():
    list_data = extract_list_file_from_path('/Users/benjamindeleener/data/data_augmentation/small')
    print 'Number of images:', len(list_data)

    patch_size = 32
    max_patches_factor = 25

    np.random.shuffle(list_data)

    for i, fname in enumerate(list_data):
        data_im, data_seg = extract_slices_from_image(list_data[i][0], list_data[i][1])
        number_of_slices = len(data_im)

        arr = range(number_of_slices)
        np.random.shuffle(arr)
        for k in arr:
            #plt.figure()
            #plt.imshow(data_im[k])
            #plt.show()

            patches = extract_patch_from_slice(data_im[k], data_seg[k], patch_size, max_patches_factor)
            number_of_patches = patches.shape[0]
            # print k, number_of_slices, number_of_patches
            for j in range(number_of_patches):
                patch_im = patches[j, :, :, 0]
                patch_seg = patches[j, :, :, 1]

                result = {}
                result['patch'] = np.expand_dims(patch_im.flatten(), axis=0)
                if patch_seg[int(patch_size / 2), int(patch_size / 2)] == 1:
                    result['class'] = 1
                else:
                    result['class'] = 0
                yield result

###############################################################################
# Main
###############################################################################
# Create the vectorizer and limit the number of features to a reasonable
# maximum
#vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, non_negative=True)

# Iterator over parsed Reuters SGML files.
data_stream = stream_images()

# We learn a binary classification between the "acq" class and all the others.
# "acq" was chosen as it is more or less evenly distributed in the Reuters
# files. For other datasets, one should take care of creating a test set with
# a realistic portion of positive instances.
all_classes = np.array([0, 1])
positive_class = 'acq'

# Here are some classifiers that support the `partial_fit` method
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    'SGD-log': SGDClassifier(loss='log'),
    'SGD-modified_huber': SGDClassifier(loss='modified_huber'),
}


def get_minibatch(patch_iter, size):
    """Extract a minibatch of examples, return a tuple X_text, y.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [(patch['patch'], patch['class']) for patch in itertools.islice(patch_iter, size)]
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    X, y = zip(*data)
    X, y = np.asarray(X, dtype=int), np.asarray(y, dtype=int)
    X = np.squeeze(X)
    return X, y


def iter_minibatches(patch_iter, minibatch_size):
    """Generator of minibatches."""
    X, y = get_minibatch(patch_iter, minibatch_size)
    while len(X):
        yield X, y
        X, y = get_minibatch(patch_iter, minibatch_size)


# test data statistics
test_stats = {'n_test': 0, 'n_test_pos': 0}

# First we hold out a number of examples to estimate accuracy
n_test_documents = 500000
tick = time.time()
X_test, y_test = get_minibatch(data_stream, n_test_documents)
parsing_time = time.time() - tick
tick = time.time()
#X_test = vectorizer.transform(X_test_text)
vectorizing_time = time.time() - tick
test_stats['n_test'] += len(y_test)
test_stats['n_test_pos'] += sum(y_test)
print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))


def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "fscore: %(fscore).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


cls_stats = {}

for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'fscore': 0.0,
             'accuracy_history': [(0, 0)], 'precision_history': [(0, 0)], 'recall_history': [(0, 0)], 'fscore_history': [(0, 0)],
             't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats

get_minibatch(data_stream, n_test_documents)
# Discard test set

# We will feed the classifier with mini-batches of 1000 documents; this means
# we have at most 1000 docs in memory at any time.  The smaller the document
# batch, the bigger the relative overhead of the partial fit methods.
minibatch_size = 10000

# Create the data_stream that parses Reuters SGML files and iterates on
# documents as a stream.
minibatch_iterators = iter_minibatches(data_stream, minibatch_size)
total_vect_time = 0.0

evaluation_factor = 10

# Main loop : iterate on mini-batchs of examples
for i, (X_train, y_train) in enumerate(minibatch_iterators):

    tick = time.time()
    #X_train = vectorizer.transform(X_train_text)
    total_vect_time += time.time() - tick

    weight_class = [sum(y_train) / float(len(y_train)), 1.0]
    sample_weights = [weight_class[sample_class] for sample_class in y_train]

    for cls_name, cls in partial_fit_classifiers.items():
        tick = time.time()
        # update estimator with examples in the current mini-batch
        #print X_train.shape, y_train.shape

        cls.partial_fit(X_train, y_train, classes=all_classes, sample_weight=sample_weights)
        y_pred = cls.predict(X_test)

        # accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time.time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        tick = time.time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        cls_stats[cls_name]['precision'] = precision_score(y_test, y_pred)
        cls_stats[cls_name]['recall'] = recall_score(y_test, y_pred)
        cls_stats[cls_name]['fscore'] = f1_score(y_test, y_pred)
        cls_stats[cls_name]['prediction_time'] = time.time() - tick
        acc_history = (cls_stats[cls_name]['accuracy'],
                       cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        precision_history = (cls_stats[cls_name]['precision'],
                            cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['precision_history'].append(precision_history)
        recall_history = (cls_stats[cls_name]['recall'],
                            cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['recall_history'].append(recall_history)
        fscore_history = (cls_stats[cls_name]['fscore'],
                            cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['fscore_history'].append(fscore_history)
        run_history = (cls_stats[cls_name]['accuracy'],
                       total_vect_time + cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        if i % evaluation_factor == 0:
            print(progress(cls_name, cls_stats[cls_name]))

    if i % evaluation_factor == 0:
        print('\n')


###############################################################################
# Plot results
###############################################################################


def plot_accuracy(x, y, x_legend, text):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification ' + text + ' as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel(text)
    plt.grid(True)
    plt.plot(x, y)

rcParams['legend.fontsize'] = 10
cls_names = list(sorted(cls_stats.keys()))

# Plot accuracy evolution
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
    accuracy, n_examples = zip(*stats['accuracy_history'])
    plot_accuracy(n_examples, accuracy, "training examples (#)", 'accuracy')
    ax = plt.gca()
    ax.set_ylim((0, 1))
plt.legend(cls_names, loc='best')

# Plot precision evolution
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
    precision, n_examples = zip(*stats['precision_history'])
    plot_accuracy(n_examples, precision, "training examples (#)", 'precision')
    ax = plt.gca()
    ax.set_ylim((0, 1))
plt.legend(cls_names, loc='best')

# Plot precision evolution
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
    fscore, n_examples = zip(*stats['fscore_history'])
    plot_accuracy(n_examples, fscore, "training examples (#)", 'fscore')
    ax = plt.gca()
    ax.set_ylim((0, 1))
plt.legend(cls_names, loc='best')

plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with runtime
    accuracy, runtime = zip(*stats['runtime_history'])
    plot_accuracy(runtime, accuracy, 'runtime (s)', 'accuracy')
    ax = plt.gca()
    ax.set_ylim((0, 1))
plt.legend(cls_names, loc='best')

# Plot fitting times
plt.figure()
fig = plt.gcf()
cls_runtime = []
for cls_name, stats in sorted(cls_stats.items()):
    cls_runtime.append(stats['total_fit_time'])

cls_runtime.append(total_vect_time)
cls_names.append('Vectorization')
bar_colors = rcParams['axes.color_cycle'][:len(cls_names)]

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                     color=bar_colors)

ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=10)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel('runtime (s)')
ax.set_title('Training Times')


def autolabel(rectangles):
    """attach some text vi autolabel on rectangles."""
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height, '%.4f' % height,
                ha='center', va='bottom')

autolabel(rectangles)
plt.show()
"""

# Plot prediction times
plt.figure()
#fig = plt.gcf()
cls_runtime = []
cls_names = list(sorted(cls_stats.keys()))
for cls_name, stats in sorted(cls_stats.items()):
    cls_runtime.append(stats['prediction_time'])
cls_runtime.append(parsing_time)
cls_names.append('Read/Parse\n+Feat.Extr.')
cls_runtime.append(vectorizing_time)
cls_names.append('Hashing\n+Vect.')
bar_colors = rcParams['axes.color_cycle'][:len(cls_names)]

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                     color=bar_colors)

ax.set_xticks(np.linspace(0.25, len(cls_names) - 0.75, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=8)
plt.setp(plt.xticks()[1], rotation=30)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel('runtime (s)')
ax.set_title('Prediction Times (%d instances)' % n_test_documents)
autolabel(rectangles)
plt.show()

"""








"""


list_data = extract_list_file_from_path('/Users/benjamindeleener/data/data_augmentation/small')
data, target = create_patches_imseg(list_data)
data = data[:15000]
target = target[:15000]
print 'Matrix shape (samples, feature):', data.shape
print 'Target shape (samples):', target.shape
"""
"""
data_centered = extract_data('/Users/benjamindeleener/data/machine_learning/scikit_learn/centered/')
target_centered = np.ones(data_centered.shape[0])
data_notcentered = extract_data('/Users/benjamindeleener/data/machine_learning/scikit_learn/not_centered/')
target_notcentered = np.zeros(data_notcentered.shape[0])
data_result = np.concatenate([data_centered, data_notcentered], axis=0)
target_result = np.concatenate([target_centered, target_notcentered], axis=0)
print 'Matrix shape (samples, feature):', data_result.shape
print 'Target shape (samples):', target_result.shape
"""
"""
n_samples = len(target)
rng = np.random.RandomState(0)

logistic = linear_model.LogisticRegression()

cv = cross_validation.ShuffleSplit(n_samples, n_iter=100, test_size=0.2, random_state=0)

title = "Learning Curves (Logistic Regression)"
plot_learning_curve(logistic, title, data, target, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
"""
"""

ss = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.25, train_size=0.1, random_state=rng)


iteration = 1
for train_index, test_index in ss:
    print 'Iteration ' + str(iteration) + ':'

    X_train = data[train_index]
    y_train = target[train_index]
    X_test = data[test_index]
    y_test = target[test_index]

    number_of_centered = np.sum(y_train)
    pourcentage_centered = 100.0 * number_of_centered / float(len(y_train))
    print 'Number of data: ', len(y_train)
    print 'Number of centered spinal cord: ' + str(round(pourcentage_centered, 2)) + '%'

    clf_result = logistic.fit(X_train, y_train)
    y_pred = logistic.predict(X_test)

    print('LogisticRegression score: %f' % clf_result.score(X_test, y_test))
    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

    iteration += 1
"""



