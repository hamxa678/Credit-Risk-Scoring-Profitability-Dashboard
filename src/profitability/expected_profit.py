from __future__ import annotations

def expected_loss(pd_value: float, exposure: float, lgd: float = 0.45) -> float:
    return float(pd_value * exposure * lgd)

def expected_profit(
    pd_value: float,
    exposure: float,
    revenue: float,
    operating_cost: float = 100.0,
    lgd: float = 0.45,
) -> float:
    return float(revenue - expected_loss(pd_value, exposure, lgd) - operating_cost)
