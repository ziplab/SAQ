from collections import defaultdict

import torch


class SAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.optimizer.param_groups[0]["params"][0].device
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            wgrads.append(torch.norm(p.grad, p=2).to(shared_device))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)
        return wgrad_norm

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]["old_p"] = p.data.clone()
            e_w = p.grad * scale.to(p)
            p.add_(e_w)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            # get back to "w" from "w + e(w)"
            p.data = self.state[p]["old_p"]
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None or "clip_value" in n:
                continue
            p.sub_(self.state[p]["eps"])
