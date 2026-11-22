from fraudlake.explain import RULES, reasons_for


def test_risky_transaction_triggers_expected_reasons():
    record = {"amt": 962.4, "amt_sum_24h": 2210.0, "amt_ratio_to_card_mean": 11.8, "is_night": 1,
              "category": "shopping_net", "secs_since_prev_txn": 1260}
    reasons = reasons_for(record)
    assert "Montant élevé (962 $)" in reasons
    assert "Transaction nocturne (22 h – 4 h)" in reasons
    assert any("11.8 fois" in r for r in reasons)
    assert any("shopping_net" in r for r in reasons)
    assert not any("Enchaînement" in r for r in reasons)


def test_ordinary_transaction_has_no_reason():
    record = {"amt": 42.0, "amt_sum_24h": 120.0, "amt_ratio_to_card_mean": 0.9, "is_night": 0,
              "category": "home", "secs_since_prev_txn": 9000}
    assert reasons_for(record) == []


def test_missing_values_are_ignored():
    assert reasons_for({}) == []


def test_rule_codes_are_unique():
    codes = [r.code for r in RULES]
    assert len(codes) == len(set(codes))
