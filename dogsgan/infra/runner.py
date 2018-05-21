from pathlib import Path
import shutil
import datetime
import hashlib
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch


from dogsgan.data.loader import create_loader


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
        self.add_image(name, image)


class TrainingRunner:
    def __init__(self, name):
        now = datetime.datetime.now()
        self.out_dir = out_root / f'{name}-{now:%Y%m%d-%H%M-%S}'
        has_cuda = torch.cuda.device_count() > 0
        if has_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        print(f'Default devlice is {self.device}')
        print(f'Output dir is {str(self.out_dir)}', flush=True)

    def run(self, epochs=10000, batch_size=128):
        self.out_dir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir=str(self.out_dir)) as writer:
            context = TrainingContext(writer)

            for e in range(epochs):
                context.epoch = e
                loader = create_loader(batch_size=batch_size)
                with tqdm(loader, desc=f'Epoch {e}') as iterable:
                    it = iter(iterable)
                    self.run_epoch(it, context)

                self.save_image_sample(context)
                self.save_snapshot(context)

    def save_image_sample(self, context):
        sample = self.sample_images()
        context.add_image_batch('Generated Images', sample)

    def save_snapshot(self, context):
        snapshot = self.out_dir / 'current.snapshot'
        snapshot_backup = self.out_dir / 'current.snapshot.backup'

        if snapshot.exists():
            shutil.move(str(snapshot), snapshot_backup)

        data = self.get_snapshot()
        torch.save({
            'epoch': context.epoch,
            'data': data
        }, snapshot)

        if snapshot_backup.exists():
            snapshot_backup.unlink()

    def run_epoch(self, it, context):
        raise NotImplementedError

    def sample_images(self):
        raise NotImplementedError

    def get_snapshot(self):
        return {}
