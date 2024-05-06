import unittest
from dms import differentiable_topk, MASK_THRESHOLD
import torch

torch.set_printoptions(precision=3)


class TestDMS(unittest.TestCase):
    def test_dms_topk(self):
        N = 10
        for a in [0, 0.3, 0.5, 0.7, 1.0]:
            c = torch.rand([N])
            a = torch.tensor([a])
            m = differentiable_topk(c, a, lambda_=c.numel())

            n_remain = (m > MASK_THRESHOLD).float().sum().int().item()
            self.assertTrue(n_remain == int((1 - a) * N))
            print(
                (
                    f"Pruning Ratio: {a}\n"
                    f"Element Importance: {c}\n"
                    f"Soft mask: {m}\n"
                    f"Number of Remained Elements: {n_remain}\n"
                ),
            )
