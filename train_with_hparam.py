#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import sys
import yaml
import smtplib
import datetime
import train_feature_extraction
import train_classification

class EmailSender(object):
    """
    Just the class to send an e-mail.
    """

    SMTP_HOST = 'smtp.gmail.com'
    SMTP_PORT = 587

    def __init__(self):
        self.server = None
        self.alias = None
        self.login = None
        self.password = None
        with open(os.path.expanduser("~/.gmail_credentials.yaml"), "r") as yaml_file:
            data = yaml.load(yaml_file)
            self.alias = data["alias"]
            self.login = data["login"]
            self.password = data["password"]
        if not self.alias or not self.login or not self.password:
            print "Can't load credentials file..."
            exit(-1)
        self.server = smtplib.SMTP()
        self.server.connect(self.SMTP_HOST, self.SMTP_PORT)
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.login, self.password)

    def __del__(self):
        if not self.server:
            self.server.quit()

    def send(self, receivers, message_subject, message_body):
        """
        ASdasd
        """
        msg = "From: " + self.alias + "<" + self.login + ">\n"
        msg = msg + "To: " + ", ".join(receivers) + "\n"
        msg = msg + "Subject: " + datetime.datetime.now().strftime('[%Y-%m-%d] %H-%M ') + message_subject + "\n\n"
        msg = msg + message_body + "\n"
        self.server.sendmail(self.login, receivers, msg)

if __name__ == "__main__":

#     #for margin in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     for margin in [0.2]:
#         for epochs in [100]:#, 100, 150, 200]:
#     #        for learning_rate in [10 ** (-i) for i in range(2, 7)]:
#             #for learning_rate in [0.01, 0.001, 0.0001, 0.00001]:
#             for learning_rate in [0.0001]:
#     #            name = "hparam_lr:" + "{0:.6f}".format(learning_rate) + "_margin:" + "{0:.2f}".format(margin)
#                 name = "hparam_margin:" + "{0:.6f}".format(margin)
#                 train_feature_extraction.train_synthetic_features_extraction(name=name, batch_size=80,
#                                                                              epochs=epochs,
#                                                                              learning_rate=learning_rate, margin=margin, gradient_clip=10.0,
#                                                                              device="/device:GPU:0")

    for epochs in [500]:#, 100, 150, 200]:
        #for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        for learning_rate in [0.001]:
            name = "class_with_perm_lr:" + "{0:.5f}".format(learning_rate)+"_500"
            train_classification.train_classification(name=name, batch_size=64, epochs=epochs,
                                                      learning_rate=learning_rate, device="/device:GPU:0",
                                                      read_block_units=[256], process_block_steps=4)

    email_sender = EmailSender()
    email_sender.send(["daniel.koguciuk@gmail.com"], "Work Done!", "")
