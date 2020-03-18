#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_lstu_common.py
# Created Date: Thursday October 31st 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 7th March 2020 6:29:14 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import  os
import  sys
import  time
import  datetime
from    functools import partial
from    data_tool.data import Celeba

import  tensorflow as tf
import  tflib as tl
import  traceback
import  imlib as im
import  numpy as np
import  pylib


class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        

    def train(self):
        
        ckpt_dir    = self.config["projectCheckpoints"]
        epoch       = self.config["totalEpoch"]
        n_d         = self.config["dStep"]
        atts        = self.config["selectedAttrs"]
        thres_int   = self.config["thresInt"]
        test_int    = self.config["sampleThresInt"]
        n_sample    = self.config["sampleNum"]
        img_size    = self.config["imsize"]
        sample_freq = self.config["sampleEpoch"]
        save_freq   = self.config["modelSaveEpoch"]
        lr_base     = self.config["gLr"]
        lrDecayEpoch= self.config["lrDecayEpoch"]
        n_att       = len(self.config["selectedAttrs"])

        if self.config["threads"] >= 0:
            cpu_config = tf.ConfigProto(intra_op_parallelism_threads = self.config["threads"]//2,
                                        inter_op_parallelism_threads = self.config["threads"]//2,
                                        device_count = {'CPU': self.config["threads"]})
            cpu_config.gpu_options.allow_growth = True
            sess = tf.Session(config=cpu_config)
        else:
            sess = tl.session()

        data_loader= Celeba(self.config["dataset_path"], self.config["selectedAttrs"], 
                            self.config["imsize"], self.config["batchSize"], part='train', 
                            sess=sess, crop=(self.config["imCropSize"]>0))

        val_loader = Celeba(self.config["dataset_path"], self.config["selectedAttrs"], 
                            self.config["imsize"], self.config["sampleNum"], part='val', 
                            shuffle=False, sess=sess, crop=(self.config["imCropSize"]>0))

        package     = __import__("components."+self.config["modelScriptName"], fromlist=True)
        GencClass   = getattr(package, 'Genc')
        GdecClass   = getattr(package, 'Gdec')
        DClass      = getattr(package, 'D')
        GP          = getattr(package, "gradient_penalty")

        package     = __import__("components.STU."+self.config["stuScriptName"], fromlist=True)
        GstuClass   = getattr(package, 'Gstu')

        Genc = partial(GencClass, dim=self.config["GConvDim"], 
                        n_layers=self.config["GLayerNum"], multi_inputs=1)

        Gdec = partial(GdecClass, dim=self.config["GConvDim"], 
                        n_layers=self.config["GLayerNum"], shortcut_layers=self.config["skipNum"],
                        inject_layers=self.config["injectLayers"], one_more_conv=self.config["oneMoreConv"])

        Gstu = partial(GstuClass, dim=self.config["stuDim"], n_layers=self.config["skipNum"], 
                        inject_layers=self.config["skipNum"], kernel_size=self.config["stuKS"], norm=None, pass_state='stu')

        D    = partial(DClass, n_att=n_att, dim=self.config["DConvDim"], 
                        fc_dim=self.config["DFcDim"], n_layers=self.config["DLayerNum"])
        
        # inputs
        
        xa  = data_loader.batch_op[0]
        a   = data_loader.batch_op[1]
        b   = tf.random_shuffle(a)
        _a  = (tf.to_float(a) * 2 - 1) * self.config["thresInt"]
        _b  = (tf.to_float(b) * 2 - 1) * self.config["thresInt"]

        xa_sample      = tf.placeholder(tf.float32, shape=[None, self.config["imsize"], self.config["imsize"], 3])
        _b_sample      = tf.placeholder(tf.float32, shape=[None, n_att])
        raw_b_sample   = tf.placeholder(tf.float32, shape=[None, n_att])
        lr             = tf.placeholder(tf.float32, shape=[])

        # generate
        z   = Genc(xa)
        zb  = Gstu(z, _b-_a)
        xb_ = Gdec(zb, _b-_a)
        with tf.control_dependencies([xb_]):
            za = Gstu(z, _a-_a)
            xa_= Gdec(za, _a-_a)

        # discriminate
        xa_logit_gan, xa_logit_att      = D(xa)
        xb__logit_gan, xb__logit_att    = D(xb_)

        wd          = tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan)
        d_loss_gan  = -wd
        gp          = GP(D, xa, xb_)
        xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)
        d_loss      = d_loss_gan + gp * 10.0 + xa_loss_att

        xb__loss_gan= -tf.reduce_mean(xb__logit_gan)
        xb__loss_att= tf.losses.sigmoid_cross_entropy(b, xb__logit_att)
        xa__loss_rec= tf.losses.absolute_difference(xa, xa_)
        g_loss      = xb__loss_gan + xb__loss_att * 10.0 + xa__loss_rec * self.config["recWeight"]

        d_var       = tl.trainable_variables('D')
        d_step      = tf.train.AdamOptimizer(lr, beta1=self.config["beta1"]).minimize(d_loss, var_list=d_var)
        g_var       = tl.trainable_variables('G')
        g_step      = tf.train.AdamOptimizer(lr, beta1=self.config["beta1"]).minimize(g_loss, var_list=g_var)

        d_summary = tl.summary({
            d_loss_gan: 'd_loss_gan',
            gp: 'gp',
            xa_loss_att: 'xa_loss_att',
        }, scope='D')

        lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')

        g_summary = tl.summary({
            xb__loss_gan: 'xb__loss_gan',
            xb__loss_att: 'xb__loss_att',
            xa__loss_rec: 'xa__loss_rec',
        }, scope='G')

        d_summary  = tf.summary.merge([d_summary, lr_summary])

        # sample
        test_label = _b_sample - raw_b_sample
        x_sample   = Gdec(Gstu(Genc(xa_sample, is_training=False),
                                test_label, is_training=False), test_label, is_training=False)

        it_cnt, update_cnt = tl.counter()

        # saver
        saver = tf.train.Saver(max_to_keep=self.config["max2Keep"])

        # summary writer
        summary_writer = tf.summary.FileWriter(self.config["projectSummary"], sess.graph)

        # initialization
        if self.config["mode"] == "finetune":
            print("Continute train the model")
            tl.load_checkpoint(ckpt_dir, sess)
            print("Load previous model successfully!")
        else:
            print('Initializing all parameters...')
            sess.run(tf.global_variables_initializer())

        # train
        try:
            # data for sampling
            xa_sample_ipt, a_sample_ipt = val_loader.get_next()
            b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
            for i in range(len(atts)):
                tmp       = np.array(a_sample_ipt, copy=True)
                tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
                tmp       = Celeba.check_attribute_conflict(tmp, atts[i], atts)
                b_sample_ipt_list.append(tmp)

            it_per_epoch = len(data_loader) // (self.config["batchSize"] * (n_d + 1))
            max_it       = epoch * it_per_epoch

            print("Start to train the graph!")
            for it in range(sess.run(it_cnt), max_it):
                with pylib.Timer(is_output=False) as t:
                    sess.run(update_cnt)

                    # which epoch
                    epoch       = it // it_per_epoch
                    it_in_epoch = it % it_per_epoch + 1
                    # learning rate
                    lr_ipt      = lr_base / (10 ** (epoch // lrDecayEpoch))

                    # train D
                    for i in range(n_d):
                        d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
                    summary_writer.add_summary(d_summary_opt, it)

                    # train G
                    g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})
                    summary_writer.add_summary(g_summary_opt, it)

                    # display
                    if (it + 1) % 100 == 0:
                        print("Epoch: (%3d) (%5d/%5d) Time: %s!" % (epoch, it_in_epoch, it_per_epoch, t))

                    # save
                    if (it + 1) % (save_freq if save_freq else it_per_epoch) == 0:
                        save_path = saver.save(sess, '%s/Epoch_(%d).ckpt'%(ckpt_dir, epoch))
                        print('Model is saved at %s!' % save_path)

                    # sample
                    if (it + 1) % (sample_freq if sample_freq else it_per_epoch) == 0:

                        x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)]
                        raw_b_sample_ipt = (b_sample_ipt_list[0].copy() * 2 - 1) * thres_int

                        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                            if i > 0:   # i == 0 is for reconstruction
                                _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
                            x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                                _b_sample: _b_sample_ipt,
                                                                                raw_b_sample: raw_b_sample_ipt}))
                            last_images = x_sample_opt_list[-1]
                            if i > 0:   # add a mark (+/-) in the upper-left corner to identify add/remove an attribute
                                for nnn in range(last_images.shape[0]):
                                    last_images[nnn, 2:5, 0:7, :] = 1.
                                    if _b_sample_ipt[nnn, i-1] > 0:
                                        last_images[nnn, 0:7, 2:5, :] = 1.
                                        last_images[nnn, 1:6, 3:4, :] = -1.
                                    last_images[nnn, 3:4, 1:6, :] = -1.
                        sample = np.concatenate(x_sample_opt_list, 2)

                        im.imwrite(im.immerge(sample, n_sample, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % \
                                    (self.config["projectSamples"], epoch, it_in_epoch, it_per_epoch))
        except:
            traceback.print_exc()
        finally:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
            print('Model is saved at %s!' % save_path)
            sess.close()