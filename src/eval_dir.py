import torch
import torch.nn.functional as F

from dice_loss import dice_coeff
from utils import BinarizeMask

def eval_dir(net, dataset, gpu=False,n_classes=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        true_mask_cline = b[2]
        true_mask = BinarizeMask(true_mask,n_classes)

        true_mask_cline = torch.from_numpy(true_mask_cline).unsqueeze(0)
        true_mask_cline = true_mask_cline.unsqueeze(0)
        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #print(true_mask.shape)
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()
            true_mask_cline = true_mask_cline.cuda()
        mask_pred = net(img)
        #print(mask_pred.shape)
        #print(true_mask.shape)
        #mask_pred = net(img)[0]
        #print(true_mask_cline.shape)

        if int(n_classes) > 1:        
            mask_pred = mask_pred * true_mask_cline
            true_mask = true_mask * true_mask_cline
        mask_pred = (mask_pred > 0.5).float()
        
        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i
