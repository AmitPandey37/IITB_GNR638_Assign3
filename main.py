# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main method to train the model."""


#!/usr/bin/python

import argparse
import json
import os
import sys
import time
import datasets
import device_utils
import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm

torch.set_num_threads(3)


def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='test_notebook')
  parser.add_argument('--dataset', type=str, default='css3d')
  parser.add_argument(
      '--dataset_path', type=str, default='../imgcomsearch/CSSDataset/output')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--embed_dim', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument(
      '--learning_rate_decay_frequency', type=int, default=9999999)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--num_iters', type=int, default=210000)
  parser.add_argument('--loss', type=str, default='soft_triplet')
  parser.add_argument('--loader_num_workers', type=int, default=4)
  parser.add_argument('--device', type=str, default='auto')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--css_query_image_mode', type=str, default='3d', choices=['2d', '3d'])
  parser.add_argument(
      '--css_target_image_mode', type=str, default='3d', choices=['2d', '3d'])
  parser.add_argument('--eval_only', action='store_true')
  parser.add_argument('--checkpoint', type=str, default='')
  parser.add_argument('--result_json', type=str, default='')
  args = parser.parse_args()
  return args


def load_dataset(opt):
  """Loads the input datasets."""
  print('Reading dataset ', opt.dataset)
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='train',
        query_image_mode=opt.css_query_image_mode,
        target_image_mode=opt.css_target_image_mode,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='test',
        query_image_mode=opt.css_query_image_mode,
        target_image_mode=opt.css_target_image_mode,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'fashion200k':
    trainset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'mitstates':
    trainset = datasets.MITStates(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.MITStates(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  else:
    print('Invalid dataset', opt.dataset)
    sys.exit()

  print('trainset size:', len(trainset))
  print('testset size:', len(testset))
  return trainset, testset


def create_model_and_optimizer(opt, texts):
  """Builds the model and related optimizer."""
  print('Creating model and optimizer for', opt.model)
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, embed_dim=opt.embed_dim)
  else:
    print('Invalid model', opt.model)
    print('available: imgonly, textonly, concat, tirg or tirg_lastconv')
    sys.exit()
  device = device_utils.resolve_device(opt.device)
  model = model.to(device)

  # create optimizer
  params = []
  # low learning rate for pretrained layers on real image datasets
  if opt.dataset != 'css3d':
    params.append({
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt.learning_rate
    })
    params.append({
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt.learning_rate
    })
  params.append({'params': [p for p in model.parameters()]})
  seen = set()
  deduped_params = []
  for group in params:
    unique_params = []
    for param in group['params']:
      if id(param) in seen:
        continue
      seen.add(id(param))
      unique_params.append(param)
    if unique_params:
      group = dict(group)
      group['params'] = unique_params
      deduped_params.append(group)
  optimizer = torch.optim.SGD(
      deduped_params,
      lr=opt.learning_rate,
      momentum=0.9,
      weight_decay=opt.weight_decay)
  return model, optimizer


def load_checkpoint(opt, model, optimizer=None):
  """Loads a checkpoint into the current model."""
  if not opt.checkpoint:
    return {}
  print('Loading checkpoint', opt.checkpoint)
  checkpoint = torch.load(
      opt.checkpoint, map_location='cpu', weights_only=False)
  state_dict = checkpoint
  if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
  model.load_state_dict(state_dict, strict=True)
  if optimizer and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if isinstance(checkpoint, dict):
    return checkpoint
  return {}


def evaluate_model(opt, model, trainset, testset):
  """Runs retrieval evaluation and optionally saves JSON metrics."""
  metrics = {}
  for split_name, dataset in [('train', trainset), ('test', testset)]:
    split_metrics = test_retrieval.test(opt, model, dataset)
    metrics[split_name] = {name: float(value) for name, value in split_metrics}
    for metric_name, metric_value in split_metrics:
      print(split_name, metric_name, round(metric_value, 4))
  if opt.result_json:
    with open(opt.result_json, 'w') as f:
      json.dump(metrics, f, indent=2, sort_keys=True)
    print('Saved metrics to', opt.result_json)
  return metrics


def train_loop(opt, logger, trainset, testset, model, optimizer, checkpoint=None):
  """Function for train loop"""
  print('Begin training')
  device = device_utils.get_module_device(model)
  losses_tracking = {}
  trainloader = trainset.get_loader(
      batch_size=opt.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=opt.loader_num_workers)
  steps_per_epoch = len(trainloader)
  if steps_per_epoch <= 0:
    raise ValueError('Training loader is empty.')
  checkpoint = checkpoint or {}
  it = int(checkpoint.get('it', 0))
  epoch = int(checkpoint.get('epoch', max(it // steps_per_epoch - 1, -1)))
  tic = time.time()
  while it < opt.num_iters:
    epoch += 1

    # show/log stats
    print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic, 4), opt.comment)
    tic = time.time()
    for loss_name in losses_tracking:
      avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
      print('    Loss', loss_name, round(avg_loss, 4))
      logger.add_scalar(loss_name, avg_loss, it)
    logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

    # test
    if epoch % 3 == 1:
      tests = []
      for name, dataset in [('train', trainset), ('test', testset)]:
        t = test_retrieval.test(opt, model, dataset)
        tests += [(name + ' ' + metric_name, metric_value)
                  for metric_name, metric_value in t]
      for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print('    ', metric_name, round(metric_value, 4))

    # save checkpoint
    torch.save({
        'it': it,
        'epoch': epoch,
        'opt': opt,
        'logdir': logger.file_writer.get_logdir(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },
               logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

    # run trainning for 1 epoch
    model.train()
    trainloader = trainset.get_loader(
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)


    def training_1_iter(data):
      assert type(data) is list
      img1 = np.stack([d['source_img_data'] for d in data])
      img1 = torch.from_numpy(img1).float().to(device)
      img2 = np.stack([d['target_img_data'] for d in data])
      img2 = torch.from_numpy(img2).float().to(device)
      mods = [str(d['mod']['str']) for d in data]

      # compute loss
      losses = []
      if opt.loss == 'soft_triplet':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=True)
      elif opt.loss == 'batch_based_classification':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=False)
      else:
        print('Invalid loss function', opt.loss)
        sys.exit()
      loss_name = opt.loss
      loss_weight = 1.0
      losses += [(loss_name, loss_weight, loss_value)]
      total_loss = sum([
          loss_weight * loss_value
          for loss_name, loss_weight, loss_value in losses
      ])
      assert not torch.isnan(total_loss)
      losses += [('total training loss', None, total_loss)]

      # track losses
      for loss_name, loss_weight, loss_value in losses:
        if loss_name not in losses_tracking:
          losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value.detach()))

      # gradient descend
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

    for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
      it += 1
      training_1_iter(data)

      # decay learing rate
      if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
        for g in optimizer.param_groups:
          g['lr'] *= 0.1

  print('Finished training')


def main():
  opt = parse_opt()
  device_utils.seed_everything(opt.seed)
  resolved_device = device_utils.resolve_device(opt.device)
  opt.device = str(resolved_device)
  print('Arguments:')
  for k in opt.__dict__.keys():
    print('    ', k, ':', str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt)
  model, optimizer = create_model_and_optimizer(opt, trainset.get_all_texts())
  checkpoint = load_checkpoint(
      opt, model, optimizer=None if opt.eval_only else optimizer)

  if opt.eval_only:
    evaluate_model(opt, model, trainset, testset)
    return

  if opt.checkpoint:
    logdir = checkpoint.get('logdir', os.path.dirname(opt.checkpoint))
    logger = SummaryWriter(logdir=logdir, purge_step=checkpoint.get('it', 0))
  else:
    logger = SummaryWriter(comment=opt.comment)
  print('Log files saved to', logger.file_writer.get_logdir())
  for k in opt.__dict__.keys():
    logger.add_text(k, str(opt.__dict__[k]))

  train_loop(opt, logger, trainset, testset, model, optimizer, checkpoint)
  logger.close()


if __name__ == '__main__':
  main()
