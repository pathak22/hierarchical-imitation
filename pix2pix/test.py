import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
import numpy as np
import cv2
from IPython import embed
import time
import sys


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # if opt.eval:
    #     model.eval()
    for i, data in enumerate(dataset):
        #import pdb; pdb.set_trace();
        if i >= opt.num_test:
            break
        # from PIL import Image
        # Image.fromarray((255.*data['A'].squeeze(0)[:3].permute([1,2,0]).data.numpy()).astype(np.uint8)).save('/nfs.yoda/gauravp/pratyusha//temp011.jpg')
        # Image.fromarray((255.*data['A'].squeeze(0)[3:6].permute([1,2,0]).data.numpy()).astype(np.uint8)).save('/nfs.yoda/gauravp/pratyusha//temp012.jpg')
        # Image.fromarray((255.*data['A'].squeeze(0)[6:].permute([1,2,0]).data.numpy()).astype(np.uint8)).save('/nfs.yoda/gauravp/pratyusha//temp013.jpg')
        
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        b = visuals['fake_B']
        im = util.tensor2im(b)
        im = im[:,:,::-1]
        # cv2.imwrite('/nfs.yoda/gauravp/pratyusha/temp/'+str(i)+'.jpg',im)
        # cv2.imwrite('/nfs.yoda/gauravp/pratyusha//temp014.jpg',im)
        # embed()
        
        # im_path = img_path[0][:-6]+str(i+opt.iii)+'.jpg'
        # print(im_path)
        # a_= cv2.imread(im_path)
        # c = np.concatenate((a_[:,0:300,:],im[:,:,0:3][:,:,::-1]),axis = 1)
        # cv2.imwrite(im_path,c)
        # time.sleep(3)
        # sys.exit()
    # save the website
    webpage.save()

    # save the website
        # b = visuals['fake_B']
        # im = util.tensor2im(b)
        # im_path = img_path[0][:-6]+str(i+36)+'.jpg'
        # a_= cv2.imread(im_path)
        # c = np.concatenate((a_[:,0:300,:],im[:,:,12:15][:,:,::-1]),axis = 1)
        # cv2.imwrite(im_path,c)
