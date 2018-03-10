from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
from tqdm import tqdm

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    # override num_images if -1 by poom (for tqdm)
    if num_images == -1:
        num_images = len(os.listdir(eval_kwargs.get('image_folder', '')))

    pbar = tqdm(total=num_images, desc='captioning..')

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    csv_entry = []
    while True:
        data = loader.get_batch(split)
        # print(data['infos'][0]['file_path'])
        n = n + loader.batch_size

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)

        # #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {}
            # entry['image_id'] = data['infos'][k]['id']
            entry['caption'] = sent
            if eval_kwargs.get('dump_path', 0) == 1:
                path = data['infos'][k]['file_path']
                # entry['file_path'] = data['infos'][k]['file_path']
                entry['file_name'] = os.path.basename(path)
                csv_entry.append([os.path.basename(path), sent])

            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            # if verbose:
            #     print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        # if verbose:
        #     print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        pbar.update(1)
    pbar.close()



    # Switch back to training mode
    model.train()
    return predictions, csv_entry
