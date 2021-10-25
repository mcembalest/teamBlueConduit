from gizmo import metrics


def mult_by_three(x):
    return x * 3


def test_mult_by_three():
    assert mult_by_three(3) == 9
    assert mult_by_three(2) != 10
