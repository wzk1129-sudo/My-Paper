"""Frozen split/config contracts; reject test/memory optimization before file access."""
import copy
import hashlib
import json
from pathlib import Path
from mayo2016_smallset_v1.runtime import ROOT, sha256
from mayo2016_smallset_v1.data import COUNTS, TRAIN

FAST_TRAIN = ('L067', 'L143')
MEMORY = ('L096', 'L109', 'L192', 'L286', 'L291', 'L310')
FAST_INDICES = tuple((i * 609 + 31) // 63 for i in range(64))
CONFIG = ROOT / 'configs/rddm_fast_dev.json'

def protocol():
    p = json.loads((ROOT / 'configs/fast_dev_protocol.json').read_text())
    assert p['fast_train'] == list(FAST_TRAIN)
    assert p['external_memory'] == list(MEMORY)
    assert set(FAST_TRAIN).isdisjoint(MEMORY)
    assert set(FAST_TRAIN) | set(MEMORY) == set(TRAIN)
    assert p['validation'] == ['L333'] and not p['test_access_permitted']
    for name, expected in p['split_sha256'].items():
        assert sha256(ROOT / 'splits' / name) == expected, 'Changed fixed split'
    assert (ROOT/'splits/fast_train_2patients.txt').read_text().splitlines() == list(FAST_TRAIN)
    assert (ROOT/'splits/external_memory_6patients.txt').read_text().splitlines() == list(MEMORY)
    assert tuple(map(int, (ROOT/'splits/L333_fast64.txt').read_text().splitlines())) == FAST_INDICES
    return p

def check_config(cfg):
    p = protocol()
    for name, expected in json.loads((ROOT/'experiments/fast_dev/formal_source_pins.json').read_text()).items():
        assert sha256(ROOT/name) == expected, 'Formal dependency changed: ' + name
    canonical = json.loads(CONFIG.read_text())
    assert cfg == canonical, 'Use the frozen fast-dev config; variants need an explicit new configuration contract'
    assert sha256(ROOT/'configs/fast_dev_protocol.json') == cfg['fast_dev']['protocol_sha256']
    formal = json.loads((ROOT/'mayo2016_smallset_v1/CONFIG.json').read_text())
    expected = copy.deepcopy(formal)
    expected['experiment_name'] = 'FoundDiff_RDDM_RLEB_fast_dev_v1'
    expected['description'] = 'Fast development screening of the fixed FoundDiff RDDM+RLEB backbone'
    expected['mode'] = 'fast_dev'
    expected['data']['train_patients'] = list(FAST_TRAIN)
    expected['training'].update(max_optimizer_steps=8000, validation_every=2000,
                                matched_checkpoint_steps=[2000,4000,6000,8000])
    expected['evaluation'].update(expected_slices=64, validation_mode='FAST',
                                  indices_file='splits/L333_fast64.txt')
    expected['safety']['maximum_steps'] = 8000
    expected['fast_dev'] = {'protocol':'configs/fast_dev_protocol.json',
        'protocol_sha256':sha256(ROOT/'configs/fast_dev_protocol.json'),
        'automatic_early_stop':False, 'external_memory_enabled':False}
    assert cfg == expected, 'Unexpected difference from formal baseline'
    return p

def validation_indices(mode):
    if mode == 'FAST': return FAST_INDICES
    if mode == 'FULL': return tuple(range(610))
    raise ValueError('Validation mode must be FAST or FULL')

def source_hashes():
    pins = json.loads((ROOT/'experiments/fast_dev/formal_source_pins.json').read_text())
    paths = list(pins) + [str(p.relative_to(ROOT)) for p in sorted((ROOT/'fast_dev').glob('*.py'))]
    paths += ['configs/fast_dev_protocol.json','configs/rddm_fast_dev.json',
              'splits/fast_train_2patients.txt','splits/external_memory_6patients.txt','splits/L333_fast64.txt']
    return {name:sha256(ROOT/name) for name in paths}

def output_directory(path):
    path = Path(path).resolve()
    root = (ROOT/'experiments/fast_dev/runs').resolve()
    if root not in path.parents: raise ValueError('Output must be a new directory under experiments/fast_dev/runs')
    path.mkdir(parents=True, exist_ok=False)
    return path
