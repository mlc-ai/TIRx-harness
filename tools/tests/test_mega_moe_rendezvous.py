from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ported.deepgemm._sm100_fp8_fp4_mega_moe import data


@pytest.fixture
def rendezvous(monkeypatch):
    state = SimpleNamespace(queues=[], ports=[], failures=[])
    available_ports = iter(range(41000, 41100))
    monkeypatch.setattr(data, "_find_free_port", lambda: next(available_ports))
    monkeypatch.setattr(data.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("MASTER_ADDR", "original-host")
    monkeypatch.setenv("MASTER_PORT", "9400")

    def make_queue():
        queue = Mock()
        queue.get.side_effect = [(rank, {"status": "OK", "rank": rank}) for rank in range(4)]
        state.queues.append(queue)
        return queue

    def spawn(function, args, nprocs, join):
        state.ports.append(data.os.environ["MASTER_PORT"])
        assert function is data._worker_entry
        assert args[:2] == ((1, 0, 3, 2), ("gpu1", "gpu0", "gpu3", "gpu2"))
        assert nprocs == 4
        assert join is True
        if state.failures:
            failure = state.failures.pop(0)
            if failure is not None:
                raise failure

    monkeypatch.setattr(
        data.mp, "get_context", lambda method: SimpleNamespace(SimpleQueue=make_queue)
    )
    monkeypatch.setattr(data.mp, "spawn", spawn)
    return state


def launch():
    return data._run_distributed(
        data.MegaMoeConfig(num_processes=4),
        "bench",
        device_indices=(1, 0, 3, 2),
        device_uuids=("gpu1", "gpu0", "gpu3", "gpu2"),
    )


def assert_cleanup(state):
    assert data.os.environ["MASTER_ADDR"] == "original-host"
    assert data.os.environ["MASTER_PORT"] == "9400"
    for queue in state.queues:
        queue.close.assert_called_once()


@pytest.mark.parametrize("message", ["EADDRINUSE", "address already in use"])
def test_distributed_port_collision_retries_on_a_new_port(rendezvous, message):
    rendezvous.failures = [data.mp.ProcessRaisedException(message, 0, 1234), None]
    assert launch() == {"status": "OK", "rank": 0}
    assert rendezvous.ports == ["41000", "41001"]
    assert_cleanup(rendezvous)


def test_distributed_non_startup_failure_is_not_retried(rendezvous):
    failure = data.mp.ProcessRaisedException("precision mismatch", 0, 1234)
    rendezvous.failures = [failure]
    with pytest.raises(data.mp.ProcessRaisedException, match="precision mismatch") as raised:
        launch()
    assert raised.value is failure
    assert len(rendezvous.ports) == 1
    assert_cleanup(rendezvous)


def test_distributed_port_collision_retries_are_bounded(rendezvous):
    failure = data.mp.ProcessRaisedException("EADDRINUSE", 0, 1234)
    rendezvous.failures = [failure] * 32
    with pytest.raises(data.mp.ProcessRaisedException, match="EADDRINUSE") as raised:
        launch()
    assert raised.value is failure
    assert len(set(rendezvous.ports)) == 32
    assert_cleanup(rendezvous)
