from whisper.utils import str2bool

def test_str2bool():
    result = str2bool("True")
    assert result