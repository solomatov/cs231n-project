import datetime
import shutil
from pathlib import Path

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

out_root = Path('out')


class TrainingContext:
    def __init__(self, writer):
        self.writer = writer
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


class TrainingRunner:
    def __init__(self, name, dataset, gen, dsc, noise_generator, use_half=False):
        self.name = name
        self.dataset = dataset
        self.noise_generator = noise_generator
        self.use_half = use_half

        now = datetime.datetime.now()
        self.out_dir = out_root / f'{name}-{now:%Y%m%d-%H%M-%S}'
        self.has_cuda = torch.cuda.device_count() > 0
        if self.has_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        print(f'Default device is {self.device}')
        print(f'Output dir is {str(self.out_dir)}', flush=True)

        self.gen = self.convert(gen)
        self.dsc = self.convert(dsc)

        self.vis_params = self.gen_noise(104)

    def run(self, epochs=10000, batch_size=128):
        self.out_dir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir=str(self.out_dir)) as writer:
            context = TrainingContext(writer)

            for e in range(epochs):
                context.epoch = e
                loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=self.has_cuda)
                with tqdm(loader, desc=f'Epoch {e}') as iterable:
                    it = (x[0] for x in iterable)
                    self.run_epoch(it, context)

                self.save_image_sample(context)
                self.save_snapshot(context, self.out_dir / 'current.snapshot')

                if e > 0 and e % 20 == 0:
                    self.save_snapshot(context, self.out_dir / f'epoch-{e:05}.snapshot')

    def save_image_sample(self, context):
        sample = self.sample_images()
        context.add_image_batch('Generated Images', sample)

    def save_snapshot(self, context, target):
        snapshot = target
        snapshot_backup = target.parent / f'{target.name}.backup'

        if snapshot.exists():
            shutil.move(str(snapshot), snapshot_backup)

        data = self.get_snapshot()
        torch.save({
            'epoch': context.epoch,
            'data': data
        }, snapshot)

        if snapshot_backup.exists():
            snapshot_backup.unlink()

    def convert(self, data):
        result = data.to(self.device)
        if self.use_half:
            result = result.half()
        return result

    def sample_images(self):
        return self.gen(self.vis_params)

    def run_epoch(self, it, context):
        raise NotImplementedError

    def gen_noise(self, n):
        return self.convert(self.noise_generator(n))

    def get_snapshot(self):
        return {}
