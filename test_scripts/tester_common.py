#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester.py
# Created Date: Wednesday February 26th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 4th March 2020 6:32:56 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
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
from    utilities.sshupload import fileUploaderClass
# from    tqdm import tqdm

class Tester(object):
    def __init__(self, config, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter

    def test(self):
        
        img_size    = self.config["imsize"]
        n_att       = len(self.config["selectedAttrs"])
        atts        = self.config["selectedAttrs"]
        thres_int   = self.config["thresInt"]
        save_dir    = self.config["projectSamples"]
        test_int    = self.config["sampleThresInt"]
        # data
        sess = tl.session()
        
        SpecifiedImages = None
        if self.config["useSpecifiedImage"]:
            SpecifiedImages = self.config["specifiedTestImages"]
        te_data = Celeba(self.config["dataset_path"] , atts, img_size, 1, 
                            part='test', sess=sess, crop=(self.config["imCropSize"]>0), im_no=SpecifiedImages)
                            
        # models
        package     = __import__(self.config["com_base"]+self.config["modelScriptName"], fromlist=True)
        GencClass   = getattr(package, 'Genc')
        GdecClass   = getattr(package, 'Gdec')
        package     = __import__(self.config["com_base"]+self.config["stuScriptName"], fromlist=True)
        GstuClass   = getattr(package, 'Gstu')
        
        Genc    = partial(GencClass, dim=self.config["GConvDim"], n_layers=self.config["GLayerNum"], 
                            multi_inputs=1)
        
        Gdec    = partial(GdecClass, dim=self.config["GConvDim"], n_layers=self.config["GLayerNum"], 
                            shortcut_layers=self.config["skipNum"], inject_layers=self.config["injectLayers"], 
                            one_more_conv=self.config["oneMoreConv"])
                            
        Gstu    = partial(GstuClass, dim=self.config["stuDim"], n_layers=self.config["skipNum"], 
                            inject_layers=self.config["stuInjectLayers"], kernel_size=self.config["stuKS"], 
                            norm=None, pass_state="stu")

        # inputs
        xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
        _b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
        raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

        # sample
        test_label = _b_sample - raw_b_sample
        x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                                test_label, is_training=False), test_label, is_training=False)
        print("Graph build success!")

        # ==============================================================================
        # =                                    test                                    =
        # ==============================================================================

        # Load pretrained model
        ckpt_dir    = os.path.join(self.config["projectCheckpoints"],self.config["ckpt_prefix"])
        restorer    = tf.train.Saver()
        restorer.restore(sess, ckpt_dir)
        print("Load pretrained model successfully!")
        # tl.load_checkpoint(ckpt_dir, sess)

        # test
        print("Start to test!")
        test_slide = False
        multi_atts = False
        
        try:
            # multi_atts = test_atts is not None
            for idx, batch in enumerate(te_data):
                xa_sample_ipt = batch[0]
                a_sample_ipt = batch[1]
                b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(n_slide if test_slide else 1)]
                # if test_slide: # test_slide
                #     for i in range(n_slide):
                #         test_int = (test_int_max - test_int_min) / (n_slide - 1) * i + test_int_min
                #         b_sample_ipt_list[i] = (b_sample_ipt_list[i]*2-1) * thres_int
                #         b_sample_ipt_list[i][..., atts.index(test_att)] = test_int
                # elif multi_atts: # test_multiple_attributes
                #     for a in test_atts:
                #         i = atts.index(a)
                #         b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]
                #         b_sample_ipt_list[-1] = Celeba.check_attribute_conflict(b_sample_ipt_list[-1], atts[i], atts)
                # else: # test_single_attributes
                for i in range(len(atts)):
                    tmp = np.array(a_sample_ipt, copy=True)
                    tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
                    tmp = Celeba.check_attribute_conflict(tmp, atts[i], atts)
                    b_sample_ipt_list.append(tmp)

                x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
                raw_a_sample_ipt = a_sample_ipt.copy()
                raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                    if not test_slide:
                        # if multi_atts: # i must be 0
                        #     for t_att, t_int in zip(test_atts, test_ints):
                        #         _b_sample_ipt[..., atts.index(t_att)] = _b_sample_ipt[..., atts.index(t_att)] * t_int
                        if i > 0:   # i == 0 is for reconstruction
                            _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int
                    x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                        _b_sample: _b_sample_ipt,
                                                                        raw_b_sample: raw_a_sample_ipt}))
                sample = np.concatenate(x_sample_opt_list, 2)

                # if test_slide:     save_folder = 'sample_testing_slide'
                # # elif multi_atts:   save_folder = 'sample_testing_multi'
                # else:              save_folder = 'sample_testing'
                # save_dir = './output/%s/%s' % (experiment_name, save_folder)
                
                # pylib.mkdir(save_dir)
                im.imwrite(sample.squeeze(0), '%s/%06d%s.png' % (save_dir,
                                                                idx + 182638 if SpecifiedImages is None else SpecifiedImages[idx], 
                                                                '_%s'%(str("jibaride")) if multi_atts else ''))

                print('%06d.png done!' % (idx + 182638 if SpecifiedImages is None else SpecifiedImages[idx]))
        except:
            traceback.print_exc()
        finally:
            sess.close()