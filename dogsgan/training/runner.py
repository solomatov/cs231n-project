import datetime
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainingContext:
    def __init__(self, writer, device):
        self.writer = writer
        self.device = device

        self.iter = 0
        self.epoch = 0

    def inc_iter(self):
        self.iter += 1

    def add_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.iter)

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
                 use_half=False, permanent_snapshot_period=20, out_path=None, args=None):
        self.name = name
        self.dataset = dataset
        self.gan_optimizer = gan_optimizer

        self.use_half = use_half
        self.permanent_snapshot_period = permanent_snapshot_period
        self.args = args
        self.start_epoch = None

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

    def run(self, epochs=10000, batch_size=128):
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
                        it = (self.convert(x[0]) for x in iterable)
                        self.gan_optimizer.run_epoch(it, ctx)

                    self.save_image_sample(ctx)
                    self.save_snapshot(ctx, self.out_dir / 'current.snapshot')

                    if e > 0 and e % self.permanent_snapshot_period == 0:
                        self.save_snapshot(ctx, self.out_dir / f'epoch-{e:05}.snapshot')
            finally:
                if self.gan_optimizer is not None:
                    self.gan_optimizer.end_training()

    def save_image_sample(self, context):
        sample = self.sample_images()
        context.add_image_batch('Generated Images', sample)

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

    def load_snapshot(self, dir):
        path = dir / 'current.snapshot'

        data = torch.load(str(path))

        self.gen = data['gen']
        self.dsc = data['dsc']
        self.start_epoch = data['epoch']
        self.out_dir = dir


    def convert(self, data):
        result = data.to(self.device)
        if self.use_half:
            result = result.half()
        return result

    def sample_images(self):
        return self.gen(self.vis_params)
