from core.utils import exact_div, str2bool

def test_exact_div():
    exact_div(4, 2)

def test_str2bool():
    result = str2bool("True")
    assert result == True
    result = str2bool("False")
    assert result == False
