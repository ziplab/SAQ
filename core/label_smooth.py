import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        num_classes,
        epsilon=0.1,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        super(LabelSmoothCrossEntropyLoss, self).__init__(
            weight, size_average, reduce, reduction
        )
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, input, target):
        logprobs = F.log_softmax(input, dim=-1)
        with torch.no_grad():
            target_probs = torch.full_like(
                logprobs, self.epsilon / (self.num_classes - 1)
            )
            target_probs.scatter_(
                dim=-1, index=target.unsqueeze(1), value=1.0 - self.epsilon
            )

        losses = -(target_probs * logprobs).sum(dim=-1)
        if self.weight is not None:
            losses = losses * self.weight
        if self.reduction == "none":
            return losses
        elif self.reduction == "sum":
            return losses.sum()
        elif self.reduction == "mean":
            return losses.mean()
        else:
            raise ValueError(
                f"The parameter 'reduction' must be in ['none','mean','sum'], bot got {self.redcution}"
            )

