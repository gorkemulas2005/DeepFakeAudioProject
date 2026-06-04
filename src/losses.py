import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    First Principles tabanlı Odaklanmış Kayıp Fonksiyonu.
    Zor örneklerin gradyan ağırlığını artırırken, kolay örnekleri baskılar.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ham logit çıktılarından log-olasılıkları hesapla
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)

        # Sadece hedef sınıfa ait olasılıkları izole et
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Matematiksel denklemin uygulanması: FL = -alpha * (1 - pt)^gamma * log(pt)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss