import torch

from logging import getLogger
logger = getLogger()


def init_paws_loss(multicrop=6, tau=0.1, T=0.25, me_max=True):
    """
    Make semi-supervised PAWS loss

    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # TODO: Step 2 should not be necessary with determined
        # Step 2: gather embeddings from all workers
        # supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        return softmax(query @ supports.T / tau) @ labels

    def loss(anchor_views, anchor_supports, anchor_support_labels, target_views, target_supports, target_support_labels, sharpen=sharpen, snn=snn):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        batch_size = len(anchor_views) // (2+multicrop)

        # Step 1: compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)
        # logger.info(f'PROBS: {probs.size()}')
        anchor_probs_max = torch.max(probs, dim=1)[0]
        # logger.info(f'MEAN-MAX-PROB: {probs_max_mean}')


        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = snn(target_views, target_supports, target_support_labels)
            # -- NOTE: added for addinition metrics
            target_max_mean = torch.mean(torch.max(targets, dim=1)[0])
            
            targets = sharpen(targets)
            # -- NOTE: added for addinition metrics
            target_sharp_max_mean = torch.mean(torch.max(targets, dim=1)[0])
            
            if multicrop > 0:
                mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
                targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            # TODO: Changed line
            # avg_probs = AllReduce.apply(torch.mean(sharpen(probs), dim=0))
            avg_probs = torch.mean(sharpen(probs), dim=0)
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))

        return loss, rloss, anchor_probs_max, target_max_mean, target_sharp_max_mean

    return loss