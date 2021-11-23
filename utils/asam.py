from collections import defaultdict

import torch


class ASAM:
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
            if "weight" in n or "clip_value" in n:
                grad = (torch.abs(p) + self.eta) * p.grad
            else:
                grad = p.grad
            wgrads.append(torch.norm(grad, p=2).to(shared_device))
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
            if "weight" in n or "clip_value" in n:
                e_w = torch.pow(p, 2) * p.grad * scale.to(p)
            else:
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
    def ascent_step_param(self, param_name):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None or n not in param_name:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if "weight" in n or "clip_value" in n:
                t_w[...] = p[...]
                # t_w + eta
                t_w.abs_().add_(self.eta)
                # t_w * grad
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.0e-16
        for n, p in self.model.named_parameters():
            if p.grad is None or n not in param_name:
                continue
            t_w = self.state[p].get("eps")
            if "weight" in n or "clip_value" in n:
                # t_w * t_w * grad
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step_param(self, param_name):
        for n, p in self.model.named_parameters():
            if p.grad is None or n not in param_name:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.zero_grad()
