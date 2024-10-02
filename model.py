#!/usr/bin/env python3

from typing import Callable

import torch
import torch.nn as nn

class Spline(nn.Module):
    def __init__(self, B: Callable, G: int) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.rand((G)))
        self.B = B

    def forward(self, x: torch.Tensor):
        # \Sigma_i c_i B_i(x)
        #print(self.c)
        return (self.c * self.B(x)).sum()

class SmallPhi(nn.Module):
    def __init__(self, B: Callable, G: int = 3) -> None:
        super().__init__()
        self.spline = Spline(B=B, G=G)
        self.b = nn.SiLU()
        self.w = nn.Parameter(torch.rand((1)))
        
        # Register spline params
        for i, param in enumerate(self.spline.parameters()):
            self.register_parameter(f"spline_c_{i}", param)
        
    def forward(self, x: torch.Tensor):
        x = (self.b(x) + self.spline(x)).sum().unsqueeze_(-1)
        x *= self.w
        return x

class KAN(nn.Module):
    def __init__(self, n: int, base: Callable, G: int = 3, device: str = None) -> None:
        super().__init__()

        self.n = n

        #self.hidden = nn.Parameter((2*n+1, n))
        self.Ws = nn.Parameter(torch.rand((n, 2*n+1))   , requires_grad=True)
        self.Cs = nn.Parameter(torch.rand((n, 2*n+1, G)), requires_grad=True)
        
        self.b = nn.SiLU()
        self.base = base

        self.device = device

    def _spline(self, c, x):
        return (c*self.base(x)).sum()

    def _apply_small_phi(self, x, i, j):
        #print("Ws:", self.Ws.shape, "Cs:", self.Cs.shape, "x:", x.shape, "i:", i, "j:", j)
        return self.Ws[i][j] * (self.b(x) + self._spline(self.Cs[i,j,:], x))

    def forward(self, x: torch.Tensor):
        out = torch.empty_like(x, device=self.device)
        
        for i in range(self.n):
            for j in range(2*self.n+1):
                out[i] = self._apply_small_phi(x[i], i, j)

        return out.sum().requires_grad_(True)


if __name__ == "__main__":
    base = torch.cos
    model = KAN(base=base, n_hidden=1)

    criterion = nn.MSELoss()

    train_data = torch.randint(0,359,(1,10000))
    train_labels = torch.cos(train_data)

    print(train_data, train_labels)
