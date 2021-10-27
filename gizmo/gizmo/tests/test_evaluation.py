from gizmo import evaluation


def add_one(x):
    return x + 1


def test_add_one():
    assert add_one(2) == 3
