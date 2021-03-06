"""
   Copyright 2018 JetBrains, s.r.o
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import datetime
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dogsgan.data.gen import generated_images_dataset
from dogsgan.data.eval import inception_score


class TrainingContext:
    def __init__(self, writer, device):
        self.writer = writer
        self.device = device

        self.iter = 0
        self.epoch = 0

    def inc_iter(self):
        self.iter += 1

    def add_scalar(self, name, value, epoch=False):
        self.writer.add_scalar(name, value, self.epoch if epoch else self.iter)

    def add_image(self, name, image):
        self.writer.add_image(name, image, self.iter)

    def add_image_batch(self, name, batch):
        image = torchvision.utils.make_grid(batch, normalize=True, scale_each=True)
        self.writer.add_image(name, image, self.epoch)

    def add_histogram(self, name, histogram):
        self.writer.add_histogram(name, histogram, self.epoch, bins='auto')


class BaseGenerator(nn.Module):
    def gen_noise(self, n):
        raise NotImplementedError

    def get_device(self):
        return next(self.parameters()).device


class GANOptimizer:
    def __init__(self):
        pass

    def start_training(self, gen, dsc):
        self.gen = gen
        self.dsc = dsc

    def end_training(self):
        pass

    def run_epoch(self, it, ctx):
        raise NotImplementedError


class TrainingRunner:
    def __init__(self, name, dataset, gen, dsc, gan_optimizer,
                 use_half=False, permanent_snapshot_period=20, out_path=None, args=None, n_class=120):
        self.name = name
        self.dataset = dataset
        self.gan_optimizer = gan_optimizer

        self.use_half = use_half
        self.permanent_snapshot_period = permanent_snapshot_period
        self.args = args
        self.start_epoch = None
        self.n_class = n_class

        now = datetime.datetime.now()

        if out_path is None:
            out_path = Path('out')

        args_str = ''
        if args is not None:
            args_str = str(args).replace(', ', ',').replace('Namespace(', '').replace(')', '')
            args_str = f'-{args_str}'

        self.out_dir = out_path / f'{name}-{now:%Y%m%d-%H%M-%S}{args_str}'
        self.has_cuda = torch.cuda.device_count() > 0
        if self.has_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        print(f'Default device is {self.device}')
        print(f'Output dir is {str(self.out_dir)}', flush=True)

        self.gen = self.convert(gen)
        self.dsc = self.convert(dsc)

        self.vis_params = self.gen.gen_noise(104)

    def train(self, epochs=10000, batch_size=128):
        self.out_dir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir=str(self.out_dir)) as writer:
            ctx = TrainingContext(writer, self.device)
            if self.gan_optimizer is not None:
                self.gan_optimizer.start_training(self.gen, self.dsc)
            try:

                if self.start_epoch is None:
                    epochs_range = range(epochs)
                else:
                    epochs_range = range(self.start_epoch, epochs)

                for e in epochs_range:
                    ctx.epoch = e
                    loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=self.has_cuda)
                    with tqdm(loader, desc=f'Epoch {e}') as iterable:
                        def one_hot(c):
                            result = torch.zeros((c.shape[0], self.n_class))
                            result[range(c.shape[0]), c] = 1.0
                            return result

                        it = ((self.convert(x[0]), one_hot(x[1])) for x in iterable)
                        self.gan_optimizer.run_epoch(it, ctx)

                    self.log_image_sample(ctx)
                    self.log_inception_score(ctx)
                    self.save_snapshot(ctx, self.out_dir / 'current.snapshot')

                    if e > 0 and e % self.permanent_snapshot_period == 0:
                        self.save_snapshot(ctx, self.out_dir / f'epoch-{e:05}.snapshot')
            finally:
                if self.gan_optimizer is not None:
                    self.gan_optimizer.end_training()

    def evaluate(self):
        size = 10000
        score, std = self.get_inception_score(size=size)
        print(f'Inception Score: {score:.3f} +- {std / (size ** 0.5):.3f}')

    def evaluate_all(self, path):
        best = None
        best_score = None
        best_std = None

        image_target = path / 'samples'
        image_target.mkdir(parents=True, exist_ok=True)

        for p in sorted(path.glob('*.snapshot')):
            print(f'evaluating {p}...')
            self.load_snapshot(p)
            score, std = self.get_inception_score(10000)
            print(f'score is {score:.3f} +- {std:.3f}')

            sample = self.sample_images()
            sample_image = torchvision.utils.make_grid(sample, normalize=True, scale_each=True)
            torchvision.utils.save_image(sample_image, str(image_target / f'{p.name}.jpeg'))

            if best is None or best_score < score:
                best = p
                best_score = score
                best_std = std

            print()

        if best is None:
            print(f'No snapshots were found')
            return

        print(f'Best model is {best}. Best score is {best_score:.3f} +- {best_std:.3f}')

    def execute(self, args):
        load_from = args.load_from
        mode = args.mode
        batch_size = args.batch_size

        if load_from is not None and mode != 'eval-all':
            self.load_snapshot(Path(load_from) / 'current.snapshot')

        if mode == 'train':
            self.train(batch_size=batch_size)
        elif mode == 'eval':
            self.evaluate()
        elif mode == 'eval-all':
            self.evaluate_all(Path(load_from))
        else:
            raise Exception(f'Unknown mode {mode}')

    def log_image_sample(self, context):
        sample = self.sample_images()
        context.add_image_batch('Generated Images', sample)

    def log_inception_score(self, context):
        score, std = self.get_inception_score()
        context.add_scalar('score/inception', score, epoch=True)
        context.add_scalar('score/inception_std', std, epoch=True)

    def get_inception_score(self, size=1000):
        self.gen.train(False)
        dataset = generated_images_dataset(self.gen, size=size)
        score, std = inception_score(dataset)
        self.gen.train(True)
        return score, std

    def save_snapshot(self, context, target):
        snapshot = target
        snapshot_backup = target.parent / f'{target.name}.backup'

        if snapshot.exists():
            shutil.move(str(snapshot), snapshot_backup)

        torch.save({
            'epoch': context.epoch,
            'gen': self.gen,
            'dsc': self.dsc
        }, snapshot)

        if snapshot_backup.exists():
            snapshot_backup.unlink()

    def load_snapshot(self, path):
        data = torch.load(str(path))

        self.gen = self.convert(data['gen'])
        self.dsc = self.convert(data['dsc'])
        self.start_epoch = data['epoch']
        self.out_dir = path.parent

    def convert(self, data):
        result = data.to(self.device)
        if self.use_half:
            result = result.half()
        return result

    def sample_images(self):
        return self.gen(self.vis_params)
