"""HFD sharing must count real processes and pending queue reservations."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from experiments.bidir import dispatch_hfd_shared as dispatcher


def test_gpu_process_limit_includes_unseen_queue_workers():
    gpus = {
        0: {'uuid': 'u0', 'pids': {1}, 'total_mb': 24000, 'used_mb': 2000},
        1: {'uuid': 'u1', 'pids': {2, 3, 4}, 'total_mb': 24000, 'used_mb': 2000},
        2: {'uuid': 'u2', 'pids': {5, 6}, 'total_mb': 24000, 'used_mb': 2000},
        3: {'uuid': 'u3', 'pids': set(), 'total_mb': 24000, 'used_mb': 23000},
    }
    states = [{'status': 'starting', 'gpu': 0, 'gpu_uuid': 'u0', 'worker': None, 'child': None},
              {'status': 'starting', 'gpu': 2, 'gpu_uuid': 'u2', 'worker': None, 'child': None}]
    assert dispatcher.suitable_gpus(gpus, states, 3, 2048) == [(2, -22000, 0, 'u0')]
    states[0]['worker'] = {'pid': 100}
    with patch.object(dispatcher, 'descendant_of', side_effect=lambda pid, identity: pid == 1):
        assert dispatcher.effective_occupancy(0, gpus[0], states) == (1, 0)


def test_tick_dispatches_only_controller_owned_hfd_jobs():
    class Store:
        def lock(self):
            return nullcontext()

        def states(self):
            return [{'id': 'other', 'status': 'pending'},
                    {'id': 'hfd', 'status': 'pending'}]

        def control(self):
            return {'paused': False}

        def job_dir(self, job):
            return Path('/unused') / job

    frozen = {'config_sha256': 'c', 'verifier_sha256': 'v',
              'implementation_sha256': {'mirepnet_ifnet': 'i'}}
    controller = {'stage': 'preflight', 'frozen': frozen,
                  'preflight_jobs': {'mirepnet_ifnet': 'hfd'}, 'formal_jobs': {}}
    spec = {'snapshot': {'kind': 'hfd_loso_004', 'pair': 'mirepnet_ifnet',
                         'config_sha256': 'c', 'verifier_sha256': 'v',
                         'implementation_sha256': 'i'}}
    gpu = {3: {'uuid': 'u3', 'pids': {31, 32}, 'total_mb': 24000, 'used_mb': 6000}}
    with patch.object(dispatcher, 'read_json', side_effect=[controller, spec]), \
            patch.object(dispatcher, 'query_gpus', return_value=gpu), \
            patch.object(dispatcher, 'dispatch_hfd') as launch:
        state, done = dispatcher.tick(Store(), Path('/controller'), 3, 2048)
    assert not done and state['scheduling'] == 'dispatched'
    assert launch.call_args.args[1]['id'] == 'hfd'
    assert launch.call_args.args[2:] == (3, 'u3')


def test_tick_respects_shared_queue_pause():
    class Store:
        def lock(self):
            return nullcontext()

        def states(self):
            return [{'id': 'hfd', 'status': 'pending'}]

        def control(self):
            return {'paused': True}

    controller = {'stage': 'preflight', 'preflight_jobs': {'pair': 'hfd'}}
    with patch.object(dispatcher, 'read_json', return_value=controller), \
            patch.object(dispatcher, 'query_gpus') as query:
        state, done = dispatcher.tick(Store(), Path('/controller'), 3, 2048)
    assert not done and state['scheduling'] == 'paused_by_shared_queue'
    query.assert_not_called()


def test_fewshot_dispatch_rejects_loso_job_identity():
    class Store:
        def lock(self):
            return nullcontext()

        def states(self):
            return [{'id': 'hfd', 'status': 'pending'}]

        def control(self):
            return {'paused': False}

        def job_dir(self, job):
            return Path('/unused') / job

    frozen = {'config_sha256': 'c', 'verifier_sha256': 'v',
              'implementation_sha256': {'mirepnet_ifnet': 'i'}}
    controller = {'stage': 'preflight', 'frozen': frozen,
                  'preflight_jobs': {'mirepnet_ifnet': 'hfd'}}
    spec = {'snapshot': {'kind': 'hfd_loso_004', 'pair': 'mirepnet_ifnet',
                         'config_sha256': 'c', 'verifier_sha256': 'v',
                         'implementation_sha256': 'i'}}
    with patch.object(dispatcher, 'read_json', side_effect=[controller, spec]), \
            patch.object(dispatcher, 'dispatch_hfd') as launch:
        try:
            dispatcher.tick(Store(), Path('/controller'), 3, 2048, 'hfd_fewshot_004')
        except ValueError:
            pass
        else:
            raise AssertionError('few-shot dispatcher must reject a LOSO job')
    launch.assert_not_called()
