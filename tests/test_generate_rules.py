from src.colocation import _generate_rules


def test_generate_rules_filters_by_threshold_and_keeps_prevalence() -> None:
    prevalent = {3: [("A", "B", "C")]}
    pi_values = {("A", "B", "C"): 0.42}
    table_instances = {
        ("A", "B", "C"): [(0, 10, 20), (1, 10, 21), (1, 11, 21)],
        ("A", "B"): [(0, 10), (1, 10), (1, 11), (2, 12)],
        ("A", "C"): [(0, 20), (1, 21), (2, 22)],
        ("B", "C"): [(10, 20), (10, 21), (11, 21), (12, 22), (13, 23)],
    }
    feature_counts = {"A": 3, "B": 4, "C": 2}

    rules = _generate_rules(
        prevalent=prevalent,
        pi_values=pi_values,
        table_instances=table_instances,
        feature_counts=feature_counts,
        min_conditional_prob=0.7,
    )

    got = {
        (
            r.antecedent,
            r.consequent,
            r.prevalence,
            round(r.conditional_probability, 5),
        )
        for r in rules
    }
    assert got == {
        (("C",), ("A", "B"), 0.42, 1.0),
        (("A", "B"), ("C",), 0.42, 0.75),
    }


def test_generate_rules_skips_single_feature_rule_when_feature_count_is_zero() -> None:
    prevalent = {2: [("A", "B")]}
    pi_values = {("A", "B"): 0.3}
    table_instances = {("A", "B"): [(0, 10), (1, 11)]}
    feature_counts = {"A": 0, "B": 2}

    rules = _generate_rules(
        prevalent=prevalent,
        pi_values=pi_values,
        table_instances=table_instances,
        feature_counts=feature_counts,
        min_conditional_prob=0.0,
    )

    assert len(rules) == 1
    assert rules[0].antecedent == ("B",)
    assert rules[0].consequent == ("A",)
    assert rules[0].conditional_probability == 1.0


def test_generate_rules_skips_multi_feature_antecedent_when_table_missing() -> None:
    prevalent = {3: [("A", "B", "C")]}
    pi_values = {("A", "B", "C"): 0.5}
    table_instances = {
        ("A", "B", "C"): [(0, 10, 20), (1, 10, 21)],
        # No size-2 table instances provided; those rules should be skipped.
    }
    feature_counts = {"A": 2, "B": 3, "C": 2}

    rules = _generate_rules(
        prevalent=prevalent,
        pi_values=pi_values,
        table_instances=table_instances,
        feature_counts=feature_counts,
        min_conditional_prob=0.0,
    )

    assert all(len(rule.antecedent) == 1 for rule in rules)

