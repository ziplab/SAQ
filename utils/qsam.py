from collections import defaultdict

import torch
import torch.nn as nn
from models.LIQ_wn_qsam import QConv2d, QLinear


class QSAM:
    def __init__(
        self,
        optimizer,
        model,
        rho=0.5,
        include_wclip=False,
        include_aclip=False,
        include_bn=True,
    ):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.include_wclip = include_wclip
        self.include_aclip = include_aclip
        self.include_bn = include_bn
        self.state = defaultdict(dict)

    @torch.no_grad()
    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.optimizer.param_groups[0]["params"][0].device
        wgrads = []
        for n, m in self.model.named_modules():
            if isinstance(m, (QConv2d, QLinear)):
                wgrads.append(torch.norm(m.x.grad, p=2).to(shared_device))

                if self.include_wclip:
                    wgrads.append(
                        torch.norm(m.weight_clip_value.grad, p=2).to(shared_device)
                    )
                if self.include_aclip and m.activation_clip_value.grad:
                    wgrads.append(
                        torch.norm(m.activation_clip_value.grad, p=2).to(shared_device)
                    )

                if hasattr(m, "bias") and m.bias is not None:
                    wgrads.append(torch.norm(m.bias.grad, p=2).to(shared_device))
            if self.include_bn:
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.grad is None:
                        continue
                    wgrads.append(torch.norm(m.weight.grad, p=2).to(shared_device))
                    wgrads.append(torch.norm(m.bias.grad, p=2).to(shared_device))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)
        return wgrad_norm

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for n, m in self.model.named_modules():
            if isinstance(m, (QConv2d, QLinear)):
                p = m.x
                self.state[m]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                m.epsilon = e_w

                if self.include_wclip:
                    p = m.weight_clip_value
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

                if self.include_aclip and m.activation_clip_value.grad:
                    p = m.activation_clip_value
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

                if hasattr(m, "bias") and m.bias is not None:
                    p = m.bias
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)
            if self.include_bn:
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.grad is None:
                        continue
                    p = m.weight
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

                    p = m.bias
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)
        self.optimizer.zero_grad()

    # @torch.no_grad()
    # def ascent_step(self):
    #     # grads = []
    #     # for n, m in self.model.named_modules():
    #     #     if isinstance(m, (QConv2d, QLinear)):
    #     #         grads.append(torch.norm(m.x.grad, p=2))

    #     #         if self.include_wclip:
    #     #             grads.append(torch.norm(m.weight_clip_value.grad, p=2))
    #     #         if self.include_aclip and m.activation_clip_value.grad:
    #     #             grads.append(torch.norm(m.activation_clip_value.grad, p=2))

    #     #         if hasattr(m, "bias") and m.bias is not None:
    #     #             grads.append(torch.norm(m.bias.grad, p=2))
    #     #     if self.include_bn:
    #     #         if isinstance(m, nn.BatchNorm2d):
    #     #             grads.append(torch.norm(m.weight.grad, p=2))
    #     #             grads.append(torch.norm(m.bias.grad, p=2))
    #     # grad_norm = torch.norm(torch.stack(grads), p=2) + 1e-12
    #     grad_norm = self._grad_norm() + 1e-12
    #     for n, m in self.model.named_modules():
    #         if isinstance(m, (QConv2d, QLinear)):
    #             eps = self.state[m].get("eps")
    #             if eps is None:
    #                 eps = torch.clone(m.x).detach()
    #                 self.state[m]["eps"] = eps
    #             eps[...] = m.x.grad[...]
    #             eps.mul_(self.rho / grad_norm)
    #             m.epsilon = eps

    #             if self.include_wclip:
    #                 eps = self.state[m].get("weight_clip_eps")
    #                 if eps is None:
    #                     eps = torch.clone(m.weight_clip_value).detach()
    #                     self.state[m]["weight_clip_eps"] = eps
    #                 eps[...] = m.weight_clip_value.grad[...]
    #                 eps.mul_(self.rho / grad_norm)
    #                 m.weight_clip_value.add_(eps)

    #             if self.include_aclip and m.activation_clip_value.grad:
    #                 eps = self.state[m].get("activation_clip_eps")
    #                 if eps is None:
    #                     eps = torch.clone(m.activation_clip_value).detach()
    #                     self.state[m]["activation_clip_eps"] = eps
    #                 eps[...] = m.activation_clip_value.grad[...]
    #                 eps.mul_(self.rho / grad_norm)
    #                 m.activation_clip_value.add_(eps)
    #             if hasattr(m, "bias") and m.bias is not None:
    #                 eps = self.state[m].get("bias_eps")
    #                 if eps is None:
    #                     eps = torch.clone(m.bias).detach()
    #                     self.state[m]["bias_eps"] = eps
    #                 eps[...] = m.bias.grad[...]
    #                 eps.mul_(self.rho / grad_norm)
    #                 m.bias.add_(eps)
    #         if self.include_bn:
    #             if isinstance(m, nn.BatchNorm2d):
    #                 eps = self.state[m].get("weight_eps")
    #                 if eps is None:
    #                     eps = torch.clone(m.weight).detach()
    #                     self.state[m]["weight_eps"] = eps
    #                 eps[...] = m.weight.grad[...]
    #                 eps.mul_(self.rho / grad_norm)
    #                 m.weight.add_(eps)

    #                 eps = self.state[m].get("bias_eps")
    #                 if eps is None:
    #                     eps = torch.clone(m.bias).detach()
    #                     self.state[m]["bias_eps"] = eps
    #                 eps[...] = m.bias.grad[...]
    #                 eps.mul_(self.rho / grad_norm)
    #                 m.bias.add_(eps)

    #     self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (QConv2d, QLinear)):
                if self.include_wclip:
                    p = m.weight_clip_value
                    p.data = self.state[p]["old_p"]
                    # m.weight_clip_value.sub_(self.state[m]["weight_clip_eps"])
                if self.include_aclip and m.activation_clip_value.grad:
                    p = m.activation_clip_value
                    p.data = self.state[p]["old_p"]

                if hasattr(m, "bias") and m.bias is not None:
                    p = m.bias
                    p.data = self.state[p]["old_p"]
            if self.include_bn:
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.grad is None:
                        continue
                    p = m.weight
                    p.data = self.state[p]["old_p"]

                    p = m.bias
                    p.data = self.state[p]["old_p"]
        self.optimizer.step()
        self.optimizer.zero_grad()

    # @torch.no_grad()
    # def descent_step(self):
    #     for n, m in self.model.named_modules():
    #         if isinstance(m, (QConv2d, QLinear)):
    #             if self.include_wclip:
    #                 m.weight_clip_value.sub_(self.state[m]["weight_clip_eps"])
    #             if self.include_aclip and m.activation_clip_value.grad:
    #                 m.activation_clip_value.sub_(self.state[m]["activation_clip_eps"])

    #             if hasattr(m, "bias") and m.bias is not None:
    #                 m.bias.sub_(self.state[m]["bias_eps"])
    #         if self.include_bn:
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.weight.sub_(self.state[m]["weight_eps"])
    #                 m.bias.sub_(self.state[m]["bias_eps"])
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()

    @torch.no_grad()
    def restore_step(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (QConv2d, QLinear)):
                if self.include_wclip:
                    p = m.weight_clip_value
                    p.data = self.state[p]["old_p"]
                    # m.weight_clip_value.sub_(self.state[m]["weight_clip_eps"])
                if self.include_aclip and m.activation_clip_value.grad:
                    p = m.activation_clip_value
                    p.data = self.state[p]["old_p"]

                if hasattr(m, "bias") and m.bias is not None:
                    p = m.bias
                    p.data = self.state[p]["old_p"]
            if self.include_bn:
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.grad is None:
                        continue
                    p = m.weight
                    p.data = self.state[p]["old_p"]

                    p = m.bias
                    p.data = self.state[p]["old_p"]
        self.optimizer.zero_grad()
