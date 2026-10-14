"""CNN + CRF for per-residue signal peptide and topology prediction.
 
The CNN decides what each residue looks like from its neighbours.
The CRF decides which label sequence is coherent, from a learned
transition matrix. Transitions are unconstrained, so whatever structure
the model ends up with was learned from the data.
"""
 
import torch
import torch.nn as nn
from torchcrf import CRF
 
from splits import LABELS, N_AA
 
 
class CnnCrf(nn.Module):
    def __init__(self, n_tokens=N_AA, n_labels=len(LABELS), dim=64,
                 dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.dilations = dilations
 
        # amino acid id -> 64 learnable numbers
        self.embed = nn.Embedding(n_tokens, dim, padding_idx=0)
 
        # padding=d keeps the length at 70; dilation=d widens the window
        self.blocks = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in dilations])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
 
        # 64 features -> 6 label scores, one position at a time
        self.head = nn.Conv1d(dim, n_labels, kernel_size=1)
 
        # 6 start + 36 transition + 6 end scores, all learned
        self.crf = CRF(n_labels, batch_first=True)
 
    @property
    def receptive_field(self):
        # kernel size 3 contributes (3-1)*d per block
        return 1 + sum(2 * d for d in self.dilations)
 
    def emissions(self, x):
        """(B, 70) ids -> (B, 70, 6) label scores."""
        h = self.embed(x)            # (B, 70, 64)
        h = h.transpose(1, 2)        # (B, 64, 70) — Conv1d wants channels in the middle
 
        for conv, norm in zip(self.blocks, self.norms):
            r = conv(h)
            r = norm(r.transpose(1, 2)).transpose(1, 2)   # normalise over channels
            h = h + self.drop(self.act(r))               # residual
 
        return self.head(h).transpose(1, 2)              # back to (B, 70, 6)
 
    def loss(self, x, y, mask):
        """CRF negative log-likelihood, averaged over the batch."""
        em = self.emissions(x)
        # y holds -100 at padding, which is not a valid state index.
        # The mask excludes those positions, but the tensor still needs legal values.
        return -self.crf(em, y.clamp(min=0), mask=mask, reduction="mean")
 
    @torch.no_grad()
    def predict(self, x, mask):
        """Viterbi decode -> (B, 70) tensor, 0 outside the mask."""
        paths = self.crf.decode(self.emissions(x), mask=mask)
        out = torch.zeros_like(x)
        for i, p in enumerate(paths):
            out[i, : len(p)] = torch.tensor(p, device=x.device)
        return out
 
    def transition_matrix(self):
        """The learned 6x6 transitions, for inspection."""
        return self.crf.transitions.detach().cpu().numpy()