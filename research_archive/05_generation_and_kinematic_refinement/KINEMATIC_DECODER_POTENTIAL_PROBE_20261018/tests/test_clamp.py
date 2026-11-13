"""限幅积分单测。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kdec.integrate import integrate_acceleration, soft_clamp_norm, H


def test_soft_clamp():
    x = torch.tensor([[3.0, 4.0]])  # norm 5
    y = soft_clamp_norm(x, 2.0)
    assert abs(torch.linalg.vector_norm(y).item() - 2.0) < 1e-5


def test_integrate_respects_vmax():
    a = torch.ones(2, 1, H, 2) * 10.0
    v0 = torch.zeros(2, 2)
    paths, v, dp, au = integrate_acceleration(a, v0, v_max=1.0, a_max=0.5)
    assert torch.linalg.vector_norm(v, dim=-1).max().item() <= 1.0 + 1e-4
    assert torch.linalg.vector_norm(au, dim=-1).max().item() <= 0.5 + 1e-4


def test_backward():
    a = torch.randn(2, 20, H, 2, requires_grad=True)
    v0 = torch.randn(2, 2)
    paths, _, _, _ = integrate_acceleration(a, v0, v_max=1.2, a_max=0.5)
    paths.pow(2).mean().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


if __name__ == "__main__":
    test_soft_clamp(); test_integrate_respects_vmax(); test_backward()
    print("ALL_CLAMP_TESTS_PASS")
