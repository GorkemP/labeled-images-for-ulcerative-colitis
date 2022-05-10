# Created by Gorkem Polat at 25.04.2022
# contact: polatgorkem@gmail.com

import torch
from torch import Tensor


class ClassDistanceWeightedLoss(torch.nn.Module):
    """
    Instead of calculating the confidence of true class, this class takes into account the confidences of
    non-ground-truth classes and scales them with the neighboring distance.
    Paper: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" (https://arxiv.org/abs/2202.05167)
    It is advised to experiment with different power terms. When searching for new power term, linearly increasing
    it works the best due to its exponential effect.

    """

    def __init__(self, class_size: int, power: float = 2., reduction: str = "mean"):
        super(ClassDistanceWeightedLoss, self).__init__()
        self.class_size = class_size
        self.power = power
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_sm = input.softmax(dim=1)

        weight_matrix = torch.zeros_like(input_sm)
        for i, target_item in enumerate(target):
            weight_matrix[i] = torch.tensor([abs(k - target_item) for k in range(self.class_size)])

        weight_matrix.pow_(self.power)

        # TODO check here, stop here if a nan value and debug it
        reverse_probs = (1 - input_sm).clamp_(min=1e-4)

        log_loss = -torch.log(reverse_probs)
        if torch.sum(torch.isnan(log_loss) == True) > 0:
            print("nan detected in forward pass")

        loss = log_loss * weight_matrix
        loss_sum = torch.sum(loss, dim=1)

        if self.reduction == "mean":
            loss_reduced = torch.mean(loss_sum)
        elif self.reduction == "sum":
            loss_reduced = torch.sum(loss_sum)
        else:
            raise Exception("Undefined reduction type: " + self.reduction)

        return loss_reduced
