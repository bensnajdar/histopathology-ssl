# taken from https://github.com/mdiephuis/SimCLR/blob/master/loss.py

import torch
import torch.nn as nn


def debug(text, tensor):
    if isinstance(tensor, torch.Tensor):
        print('{}\n\tis nan: {}\n\tis inf: {}\n\tmax: {}\n\tmin: {}'.format(
            text,
            torch.any(torch.isnan(tensor)),
            torch.any(torch.isinf(tensor)),
            torch.max(tensor),
            torch.min(tensor)
        ))
    else:
        for i, t in enumerate(tensor):
            debug(f'{text} - {i}', t)


class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):
        x = torch.cat((xi, xj), dim=0)

        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat2 = torch.exp(sim_mat / self.tau)

        # no diag because it's not differentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat2 = sim_mat2.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda()

        # loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat2, dim=-1) - norm_sum)))

        a = torch.sum(sim_mat2, dim=-1)
        b = (a - norm_sum)
        c = sim_match / b
        d = -torch.log(c)
        loss = torch.mean(d)

        return loss
