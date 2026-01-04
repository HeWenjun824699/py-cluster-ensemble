import pytest
import numpy as np
from pce.metrics import evaluation_single, evaluation_batch


def test_evaluation_single_perfect():
    Y = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 0, 1, 1, 2, 2])
    
    res = evaluation_single(y, Y, time=1.0)
    
    assert res['ACC'] == pytest.approx(1.0)
    assert res['NMI'] == pytest.approx(1.0)
    assert res['AR'] == pytest.approx(1.0)
    assert res['Time'] == 1.0


def test_evaluation_single_imperfect():
    Y = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 1, 1, 2, 2, 0])  # Mixed up
    
    res = evaluation_single(y, Y)
    
    assert res['ACC'] < 1.0
    assert res['Time'] is None


def test_evaluation_batch():
    Y = np.array([0, 0, 1, 1])
    y1 = np.array([0, 0, 1, 1])
    y2 = np.array([0, 1, 0, 1])
    
    labels_list = [y1, y2]
    time_list = [0.1, 0.2]
    
    res_list = evaluation_batch(labels_list, Y, time_list=time_list)
    
    assert len(res_list) == 2
    assert res_list[0]['ACC'] == 1.0
    assert res_list[0]['Time'] == 0.1
    assert res_list[1]['ACC'] < 1.0
    assert res_list[1]['Time'] == 0.2
