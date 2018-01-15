#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import train

if __name__ == "__main__":
    
    for learning_rate in [10 ** (-i) for i in range(2, 6)]:
        for margin in [0.1, 0.2, 0.3]:
            name = "hparam_lr:" + "{0:.5f}".format(learning_rate) + "_margin:" + "{0:.1f}".format(margin)
            train.train_pointnet(name=name, batch_size=16, epochs=100, learning_rate=0.001, margin=0.2)
