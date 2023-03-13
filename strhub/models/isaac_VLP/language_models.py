import numpy as np
import torch
import torch.nn.functional as F


def ReplacedTokenDetection_Loss(gt, logits_inter, rtd_logits, pad_id=96):
    padding_mask = gt == pad_id
    
    rtd_target = (gt == pred_inter).to(torch.int)
    rtd_target = (padding_mask * pad_id).to(torch.int) + ~padding_mask & rtd_target
    import ipdb; ipdb.set_trace(context=11) # #FF0000
    loss = F.cross_entropy(rtd_logits.flatten(end_dim=1), rtd_target.flatten(), ignore_index=pad_id)
    return loss


def LanguageModeling_Loss(gt, logits, pad_id=96):
    loss = F.cross_entropy(logits.flatten(end_dim=1), gt.flatten(), ignore_index=pad_id)
    return loss


def CorrectiveLanguageModeling_Loss(gt, logits_inter, logits, rtd_logits, pad_id=96, rtd_weight=50):
    """Simplified Corrective Language Model.
    Includes RTD task, and prediction of correct tokens at all positions.

    Args:
        gt : ground truth token indices, Shape : N x S
        logits_inter : logits of intermediate prediction from decoder, Shape : N x S
        logits : logits of final prediction from refiner Shape : N x S
        pad_id : index value of [PAD] token
    """
    rtd_loss = ReplacedTokenDetection_Loss(gt, logits_inter, rtd_logits, pad_id=pad_id)
    lm_loss = LanguageModeling_Loss(gt, logits, pad_id=pad_id)
    clm_loss = lm_loss + rtd_weight * rtd_loss
    return clm_loss
    

if __name__ == '__main__':
    gt = torch.IntTensor([[95,2,3,4,0,96,96,96], [95,1,3,0,96,96,96,96], [95,2,0,96,96,96,96,96]])
    pred_inter = torch.IntTensor([[95,5,6,4,0,96,96,96], [95,1,9,0,96,96,96,96], [95,2,0,96,96,96,96,96]])
    pred = torch.IntTensor([[95,2,3,4,0,96,96,96], [95,1,3,0,96,96,96,96], [95,2,0,96,96,96,96,96]])
    CorrectiveLanguageModeling_Loss(gt, pred_inter, pred)