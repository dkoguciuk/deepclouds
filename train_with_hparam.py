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
    
    for margin in [0.1, 0.2, 0.3]:
        for learning_rate in [10 ** (-i) for i in range(2, 7)]:
            name = "hparam_lr:" + "{0:.6f}".format(learning_rate) + "_margin:" + "{0:.1f}".format(margin)
            train_synthetic.train_synthetic(name=name, batch_size=32, epochs=10000, learning_rate=learning_rate, margin=margin, device="/device:GPU:0")
