#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import train_synthetic

if __name__ == "__main__":

#    for margin in [0.01, 0.05, 0.1, 0.2, 0.5]:
    for epochs in [50, 100, 150, 200]:
#        for learning_rate in [10 ** (-i) for i in range(2, 7)]:
        for learning_rate in [0.0001]:
#            name = "hparam_lr:" + "{0:.6f}".format(learning_rate) + "_margin:" + "{0:.2f}".format(margin)
            name = "hparam_lr:" + "{0:.6f}".format(learning_rate) + "_epochs:" + "{0:d}".format(epochs)
            train_synthetic.train_synthetic_features_extraction(name=name, batch_size=80,
                                                                epochs=epochs,
                                                                learning_rate=learning_rate, margin=0.2,
                                                                device="/device:GPU:0")
