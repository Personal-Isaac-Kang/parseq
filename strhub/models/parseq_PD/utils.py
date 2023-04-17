import numpy as np
import torch
import torch.nn.functional as F

class ParallelDecoding():
    """Mask related functionalities for Scheduled Parallel Decoding. """
    def __init__(self, max_iter, num_tokens, batch_size, schedule='cosine'):
        self.T = max_iter
        self.N = num_tokens
        self.bs = batch_size
        self.mask_schedule = schedule

    def mask_scheduling_function(self, r):
        """Converts [0, 1] to masking ratio """
        assert r >= 0 and r <= 1
        if self.mask_schedule == 'cosine':
            return np.cos(np.deg2rad(90 * r))
        else:
            raise NotImplementedError
        
    def get_number_of_masked_tokens(self, r):
        return int(np.floor(self.mask_scheduling_function(r) * self.N))
    
    def get_k(self, t):
        """Get number of token to unmask at given timestep """
        prev_n = self.get_number_of_masked_tokens((t - 1) / self.T)
        cur_n = self.get_number_of_masked_tokens(t / self.T)
        assert prev_n >= cur_n
        return prev_n - cur_n
    
    def get_random_mask(self):
        """Get random mask for training"""
        r = torch.rand(1)
        n = self.get_number_of_masked_tokens(r)
        mask = torch.zeros((self.bs, self.N), dtype=torch.bool)
        mask[:, :n] = True
        mask = mask[:, torch.randperm(self.N)]
        L_mask = F.pad(mask, [1, 0], "constant", 0)
        O_mask = mask
        return L_mask.detach(), O_mask.detach(), r, n
        
    def decode_step(self, t, logits, mask):
        assert t > 0
        conf = logits.softmax(-1).max(-1)[0]
        conf = conf.masked_fill(~mask, 1.0)
        conf_sorted, sort_indices = torch.sort(conf)
        n_prev = self.get_number_of_masked_tokens(t - 1)
        n = self.get_number_of_masked_tokens(t)
        
        
        
        mask = self.get_mask(t, mask)