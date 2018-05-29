from dogsgan.training.runner import GANOptimizer, TrainingContext

import torch
import torch.optim as optim
import torch.nn.functional as F


class VanillayGANOptimizer(GANOptimizer):
    def __init__(self, dsc_lr=2e-4, gen_lr=2e-4, betas=(0.5, 0.9)):
        super().__init__()

        self.dsc_lr = dsc_lr
        self.gen_lr = gen_lr
        self.betas = betas

    def start_training(self, gen, dsc):
        super().start_training(gen, dsc)

        self.dsc_opt = optim.Adam(self.dsc.parameters(), lr=self.dsc_lr, betas=self.betas)
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=self.gen_lr, betas=self.betas)

    def run_epoch(self, it, ctx):
        for X_real in it:
            ctx.inc_iter()

            self.dsc.zero_grad()

            n = X_real.shape[0]
            X_fake = self.gen(ctx.gen_noise(n))
            y_fake = torch.zeros((n, 1), device=ctx.device)
            y_real = torch.ones((n, 1), device=ctx.device)

            y = torch.cat([y_real, y_fake])
            y_ = torch.cat([self.dsc(X_real), self.dsc(X_fake)])

            dsc_loss = F.binary_cross_entropy(y_, y)
            dsc_loss.backward()
            self.dsc_opt.step()

            self.dsc.zero_grad()
            self.gen.zero_grad()
            X_fake = self.gen(ctx.gen_noise(n))
            y_ = self.dsc(X_fake)

            gen_loss = -torch.mean(torch.log(y_))
            gen_loss.backward()
            self.gen_opt.step()

            ctx.add_scalar('loss/gen', gen_loss)
            ctx.add_scalar('loss/dsc', dsc_loss)


class WGANOptimizer(GANOptimizer):
    def __init__(self, lr=5e-5):
        super().__init__()

        self.lr = lr

    def start_training(self, gen, dsc):
        super().start_training(gen, dsc)



    def end_training(self):
        super().end_training()


