from dogsgan.training.runner import GANOptimizer

import torch
import torch.optim as optim
import torch.nn.functional as F


class VanillaGANOptimizer(GANOptimizer):
    def __init__(self, dsc_lr=2e-4, gen_lr=2e-4, betas=(0.5, 0.9), loss='ls'):
        super().__init__()

        self.dsc_lr = dsc_lr
        self.gen_lr = gen_lr
        self.betas = betas

        if loss == 'bce':
            self.dsc_loss_fn = binary_cross_entropy
            self.gen_loss_fn = lambda x: -torch.log(x)
        elif loss == 'ls':
            self.dsc_loss_fn = least_squares_loss
            self.gen_loss_fn = lambda x: 0.5 * (x - 1) ** 2
        else:
            raise Exception(f'Unknown loss function {loss}')

    def start_training(self, gen, dsc):
        super().start_training(gen, dsc)

        self.dsc_opt = optim.Adam(self.dsc.parameters(), lr=self.dsc_lr, betas=self.betas)
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=self.gen_lr, betas=self.betas)

    def run_epoch(self, it, ctx):
        for X_real in it:
            ctx.inc_iter()

            self.dsc.zero_grad()

            n = X_real.shape[0]
            X_fake = self.gen(self.gen.gen_noise(n))
            y_fake = torch.zeros((n, 1), device=ctx.device)
            y_real = torch.ones((n, 1), device=ctx.device)

            y = torch.cat([y_real, y_fake])
            scores = torch.cat([self.dsc(X_real), self.dsc(X_fake)])
            dsc_loss = self.dsc_loss_fn(scores, y).mean()

            dsc_loss.backward()
            self.dsc_opt.step()

            self.dsc.zero_grad()
            self.gen.zero_grad()
            X_fake = self.gen(self.gen.gen_noise(n))
            y_ = self.dsc(X_fake)

            gen_loss = torch.mean(self.gen_loss_fn(F.sigmoid(y_)))
            gen_loss.backward()
            self.gen_opt.step()

            ctx.add_scalar('loss/gen', gen_loss)
            ctx.add_scalar('loss/dsc', dsc_loss)

            ctx.add_scalar('param/gen', param_norm(self.gen))
            ctx.add_scalar('param/dsc', param_norm(self.dsc))


class WGANOptimizer(GANOptimizer):
    def __init__(self, lr=5e-5, n_dsc=5):
        super().__init__()

        self.lr = lr
        self.n_dsc = n_dsc

    def start_training(self, gen, dsc):
        super().start_training(gen, dsc)
        self.dsc_opt = optim.RMSprop(self.dsc.parameters(), lr=self.lr)
        self.gen_opt = optim.RMSprop(self.gen.parameters(), lr=self.lr)

    def run_epoch(self, it, ctx):
        while True:
            try:
                for _ in range(self.n_dsc):
                    X_real = next(it)
                    self.dsc.zero_grad()
                    self.gen.zero_grad()
                    n = X_real.shape[0]
                    noise = self.gen.gen_noise(n)
                    X_fake = self.gen(noise)
                    critic_f = self.dsc(X_fake)
                    critic_r = self.dsc(X_real)
                    critic_loss = torch.mean(critic_f) - torch.mean(critic_r)

                    critic_loss.backward()

                    critic_grad = grad_norm(self.dsc)

                    self.dsc_opt.step()
                    self.dsc.clip()

                self.dsc.zero_grad()
                self.gen.zero_grad()
                noise = self.gen.gen_noise(X_real.shape[0])
                gen_loss = -torch.mean(self.dsc(self.gen(noise)))
                gen_loss.backward()

                gen_grad = grad_norm(self.gen)

                self.gen_opt.step()

                ctx.add_scalar('loss/critic', critic_loss)
                ctx.add_scalar('loss/gen', gen_loss)

                ctx.add_scalar('grad/critic', critic_grad)
                ctx.add_scalar('grad/gen', gen_grad)

                ctx.inc_iter()
            except StopIteration:
                break


class WGANGPOptimizer(GANOptimizer):
    def __init__(self, lr=1e-4, betas=(0.5, 0.9), l=10.0, n_dsc=5):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.l = l
        self.n_dsc = n_dsc

    def start_training(self, gen, dsc):
        super().start_training(gen, dsc)
        self.dsc_opt = optim.Adam(self.dsc.parameters(), lr=self.lr, betas=self.betas)
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=self.lr, betas=self.betas)

    def run_epoch(self, it, ctx):
        while True:
            try:
                for _ in range(self.n_dsc):
                    X_real = next(it)

                    n = X_real.shape[0]
                    self.dsc.zero_grad()
                    self.gen.zero_grad()
                    noise = self.gen.gen_noise(n)
                    X_fake = self.gen(noise)
                    critic_f = self.dsc(X_fake)
                    critic_r = self.dsc(X_real)

                    a = torch.zeros([1, 1, 1, 1], device=ctx.device).uniform_()

                    X_mix = X_real * a + X_fake * (1.0 - a)
                    gp_grad = torch.autograd.grad(outputs=self.dsc(X_mix), inputs=X_mix,
                                                  grad_outputs=torch.ones([n, 1], device=ctx.device),
                                                  retain_graph=True, create_graph=True)[0]

                    gp = torch.mean((gp_grad.norm(dim=1).norm(dim=1).norm(dim=1) - 1) ** 2)
                    critic_loss = torch.mean(critic_f) - torch.mean(critic_r) + self.l * gp

                    critic_loss.backward()

                    critic_grad = grad_norm(self.dsc)

                    self.dsc_opt.step()

                self.dsc.zero_grad()
                self.gen.zero_grad()
                noise = self.gen.gen_noise(n)
                gen_loss = -torch.mean(self.dsc(self.gen(noise)))
                gen_loss.backward()

                gen_grad = grad_norm(self.gen)

                self.gen_opt.step()

                ctx.add_scalar('loss/critic', critic_loss)
                ctx.add_scalar('loss/gen', gen_loss)

                ctx.add_scalar('grad/critic', critic_grad)
                ctx.add_scalar('grad/gen', gen_grad)
                ctx.add_scalar('grad/gp', gp)

                ctx.inc_iter()
            except StopIteration:
                break


def grad_norm(m):
    return max(p.grad.abs().max() for p in m.parameters())


def param_norm(m):
    return max(p.abs().max() for p in m.parameters())


def binary_cross_entropy(scores, target):
    # a trick to implement a binary cross entropy loss mentioned in the GAN problem in HW3
    return scores.clamp(min=0) - scores * target + (1 + (-scores.abs()).exp()).log()


def least_squares_loss(scores, target):
    scores = F.sigmoid(scores)
    return 0.5 * (target * (scores - 1) ** 2 + (1 - target) * (scores) ** 2)