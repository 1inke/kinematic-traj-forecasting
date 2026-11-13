"""积分器单元测试：恒加速解析解 + 反传。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kdec.integrate import H, DT, integrate_acceleration, constant_accel_closed_form, initial_velocity_from_past


def test_constant_accel_matches_closed_form():
    torch.manual_seed(0)
    B = 5
    a = torch.randn(B, 2)
    v0 = torch.randn(B, 2)
    aa = a[:, None, None, :].expand(B, 1, H, 2).clone()
    paths, v, dp = integrate_acceleration(aa, v0)
    closed = constant_accel_closed_form(a, v0)
    err = (paths[:, 0] - closed).abs().max().item()
    assert err < 1e-5, err
    # velocity last: v0 + H*dt*a
    v_last = v0 + H * DT * a
    assert (v[:, 0, -1] - v_last).abs().max().item() < 1e-5


def test_zero_accel_constant_velocity():
    B = 3
    v0 = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-0.5, 0.5]])
    a = torch.zeros(B, 1, H, 2)
    paths, v, dp = integrate_acceleration(a, v0)
    t = torch.arange(1, H + 1).float()
    expect = t[None, :, None] * DT * v0[:, None, :]
    assert (paths[:, 0] - expect).abs().max().item() < 1e-6
    assert (v[:, 0] - v0[:, None, :]).abs().max().item() < 1e-6


def test_initial_velocity_from_past():
    past = torch.zeros(2, 8, 2)
    past[:, -2] = torch.tensor([[0.4, 0.0], [0.0, -0.8]])
    # p0=0, v0 = -past[-2]/dt
    v0 = initial_velocity_from_past(past)
    assert torch.allclose(v0, torch.tensor([[-1.0, 0.0], [0.0, 2.0]]))


def test_backward_through_integration():
    a = torch.randn(4, 20, H, 2, requires_grad=True)
    v0 = torch.randn(4, 2)
    paths, _, _ = integrate_acceleration(a, v0)
    loss = paths.pow(2).mean()
    loss.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().sum().item() > 0


if __name__ == "__main__":
    test_constant_accel_matches_closed_form()
    test_zero_accel_constant_velocity()
    test_initial_velocity_from_past()
    test_backward_through_integration()
    print("ALL_UNIT_TESTS_PASS")
