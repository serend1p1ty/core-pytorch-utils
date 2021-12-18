from cpu.history_buffer import HistoryBuffer


def test_history_buffer():
    his_buf = HistoryBuffer(window_size=4)
    his_buf.update(0.1)
    his_buf.update(0.2)
    his_buf.update(0.5)
    his_buf.update(0.8)
    assert his_buf.avg == 0.4
    assert his_buf.global_avg == 0.4
    his_buf.update(1.0)
    assert his_buf.avg == 0.625
    assert his_buf.global_avg == 0.52
