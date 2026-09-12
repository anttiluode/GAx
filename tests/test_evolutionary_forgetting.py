from experiments.gate2_evolutionary_forgetting import modal_forgetting_receipt


def test_modal_forgetting_matches_spectrum():
    receipt = modal_forgetting_receipt(second_height=0.93, generations=40)
    assert receipt["max_prediction_error"] < 1e-12
    assert 4.0 < receipt["half_life_generations"] < 6.0


def test_near_tied_peak_preserves_mode_much_longer():
    fast = modal_forgetting_receipt(second_height=0.93, generations=20)
    slow = modal_forgetting_receipt(second_height=0.99, generations=20)
    assert slow["half_life_generations"] > 5.0 * fast["half_life_generations"]
    assert slow["spectral_ratio"] > fast["spectral_ratio"]
    assert slow["max_prediction_error"] < 1e-11
