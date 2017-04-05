import numpy as np
from sklearn.metrics.pairwise import *
from sklearn.svm import SVC
import functools
from typing import List

class Source(object):
    def __init__(self):
        self.X_train = np.array()
        self.Y_train = np.array()
        self.X_test = np.array()
        self.y_test = np.array()




def generate_kernel(X_train, X_test, kernel_name, option):
    """X_train
    X_test
    type: 'rbf', 'poly', etc,
        sklearn.metrics.pairwise.polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1)
        sklearn.metrics.pairwise.rbf_kernel(X, Y=None, gamma=None)
    option: a dictionary

    return a pair for K_train and  K_test"""
    m, _ = X_train.shape
    X_all = np.vstack(X_train, X_test)
    if kernel_name == 'rbf':
        K = rbf_kernel(X_all, X_train, **option)
    elif kernel_name == 'poly':
        K = polynomial_kernel(X_all, X_train, **option)
    else:
        raise Exception('Kernel {} is not implemented'.format(kernel_name))
    K_train, K_test = K[0:m, :], K[m:-1, :]
    return (K_train, K_test)





def product_combination(kernel_list_iterable):
    for ind, ker in enumerate(kernel_list_iterable):
        if ind == 0:
            K = ker
        K *= ker
    return K

def generate_linear_combination(kernel_list_iterable, weights):
    # used generator is the preferred method since it is korenmemory efficient
    for ind, ker in enumerate(kernel_list_iterable):
        if ind == 0:
            K = ker * weights[ind]
        K += ker * weights[ind]
    return K

def sum_combination(kernel_list_iterable):
    for ind, ker in enumerate(kernel_list_iterable):
        if ind == 0:
            K = ker
        K += ker
    return K


def cal_accuracy_based_kernels(kernel_train_list, kernel_test_list, accuracy, cut_off=0.5):
    accuracy = np.array(list(accuracy))  # to handle both np.array and other iterables
    accuracy = accuracy - cut_off
    weights = accuracy / sum(accuracy)
    print('combine kernels with weights {}'.format(weights))
    return lambda ker_list: generate_linear_combination(ker_list, weights)


class KernelMachine(object):
    # must set kernel generation methods first
    # learn kernel combination after input_data has been ran


    def input_data(self, sources):
        first_source = sources[0]
        self.sources = sources
        self.n_train, m = first_source.X_train.shape
        self.n_test = first_source.y_test.shape
        self.n_all = self.n_train + self.n_test
        self.kernel_funcs = []
        self.y_train = first_source.Y_train
        self.y_test = first_source.Y_test


    def set_kernel_gen_methods(self, kernel_funcs):
        """kernel funcs, each kernel func should do: X, Y -> N_x, N_y kernel"""
        self.kernel_funcs = kernel_funcs

    def set_kernel_combination_rules(self, comb_func):
        self.comb_func = comb_func

    def use_heruistic_similarity(self):
        weights = []
        y_h = self.y_train.reshape(1, self.n_train)
        y_v = self.y_train.reshape(self.n_train, 1)
        train_index = range(self.n_train)
        for ind in range(len(self.sources)):
            K = self.get_ith_kernel(ind, train_index, train_index)
            cur_weight =  np.dot(y_h, np.dot(K, y_v))/ sum(sum(K * K))
            weights.append(cur_weight)
        weights = np.array(weights)
        norm_weights = weights / sum(weights)
        print('combine kernels with weights {}'.format(norm_weights))
        self.comb_func =  lambda some_iterable: generate_linear_combination(some_iterable, weights)


    def get_ith_kernel(self, index, X1_indices, X2_indices):
        cur_source = self.sources[index]
        train_X, test_X = cur_source.train_X, cur_source.train_Y
        all_X = np.vstack(train_X, test_X)
        X1, X2 = all_X[X1_indices, :], all_X[X2_indices]
        cur_func = self.kernel_funcs[index]
        return cur_func(X1, X2)

    def get_all_kernels(self, X1_indices, X2_indices):
        # return an iterable to yield kernel one by one to save memory
        for ind in range(len(self.sources)):
            yield self.get_ith_kernel(ind, X1_indices, X2_indices)

    def get_combined_kernel(self, X1_indices, X2_indices):
        self.comb_func(self.get_all_kernels(X1_indices, X2_indices))


    def fit(self, options):
        self.clf = SVC(kernel = lambda indices: self.get_combined_kernel(indices, range(self.n_train)))
        self.clf(range(self.n_train), self.y_train, **options)

    def predict_svm(self):
        return self.clf.predict(range(self.n_train, self.n_all))
