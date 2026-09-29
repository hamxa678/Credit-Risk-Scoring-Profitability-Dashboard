from src.profitability.expected_profit import expected_loss, expected_profit

def test_expected_loss():
    assert expected_loss(0.1, 10000, 0.5) == 500

def test_expected_profit():
    assert expected_profit(0.1, 10000, 1500, 100, 0.5) == 900
