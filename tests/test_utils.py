from pce.utils.show_function_params import show_function_params


def test_show_function_params(capsys):
    # Test if it prints something
    show_function_params('cspa')
    captured = capsys.readouterr()
    assert "Parameter Status" in captured.out
    assert "Parameters" not in captured.out  # The function prints "Parameter Status", not docstring

    # Test invalid method
    show_function_params('invalid_method')
    captured = capsys.readouterr()
    assert "[Error]" in captured.out
