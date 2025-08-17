from whisper import utils

def test_exact_div():
    utils.exact_div(4, 2)

def test_str2bool():
    result = utils.str2bool("True")
    assert result == True
    result = utils.str2bool("False")
    assert result == False
