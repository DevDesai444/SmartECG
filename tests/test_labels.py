from smartecg.data.labels import codes_to_labels, parse_scp, CLASSES


def test_normal_mapping():
    y = codes_to_labels({"NORM": 100.0})
    assert y[CLASSES.index("normal")] == 1.0
    assert y.sum() == 1.0


def test_af_includes_aflt():
    y = codes_to_labels({"AFLT": 80.0})
    assert y[CLASSES.index("af")] == 1.0


def test_stemi_codes_trigger_stemi_only():
    y = codes_to_labels({"AMI": 100.0})
    assert y[CLASSES.index("stemi")] == 1.0
    # not a conduction or arrhythmia label
    assert y[CLASSES.index("conduction")] == 0.0
    assert y[CLASSES.index("arrhythmia")] == 0.0


def test_below_threshold_dropped():
    y = codes_to_labels({"AFIB": 30.0}, threshold=50.0)
    assert y.sum() == 0.0


def test_multilabel_coexist():
    # AF + LBBB → AF and conduction both set
    y = codes_to_labels({"AFIB": 100.0, "CLBBB": 100.0})
    assert y[CLASSES.index("af")] == 1.0
    assert y[CLASSES.index("conduction")] == 1.0


def test_parse_scp_string():
    s = "{'NORM': 100.0, 'SR': 0.0}"
    d = parse_scp(s)
    assert d["NORM"] == 100.0
