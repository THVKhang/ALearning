import torch.nn as nn


class MultimodalJointLoss(nn.Module):
    """Joint classification loss combining main fused cross-entropy with optional
    auxiliary per-modality cross-entropy supervision (`aux_weight`).
    """

    def __init__(self, aux_weight=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        loss = self.ce(outputs["logits"], targets)
        if self.aux_weight > 0.0 and "logits_rgb" in outputs and "logits_depth" in outputs:
            aux_rgb = self.ce(outputs["logits_rgb"], targets)
            aux_depth = self.ce(outputs["logits_depth"], targets)
            loss = loss + self.aux_weight * (aux_rgb + aux_depth)
        return loss
