"""Guard against the mixed-budget reporting failure that prompted this replay."""
import json
from pathlib import Path

import pytest
import yaml

from experiments.bidir.budget100 import CONFIGS, FAMILIES, ROOT, epoch_count, sha, verify_completion


def test_all_replay_configs_have_an_explicit_100_epoch_budget():
    paths=list(CONFIGS.glob('*.yaml'))
    assert len(paths)>=12
    for path in paths:
        cfg=yaml.safe_load(path.read_text())
        assert cfg['joint_epochs']==100,path
        if 'epochs' in cfg: assert cfg['epochs']==100,path
        for value in cfg.get('teacher_training',{}).values():
            if isinstance(value,dict) and 'epochs' in value: assert value['epochs']==100,path
        for value in cfg.get('pairs',{}).values():
            if 'epochs' in value: assert value['epochs']==100,path


def test_auxiliary_loss_settings_are_not_changed_by_the_epoch_replay():
    for _,filename,_ in FAMILIES.values():
        if filename is None:continue
        old=yaml.safe_load((ROOT/'configs/experiments'/filename).read_text())
        new=yaml.safe_load((CONFIGS/filename).read_text())
        for field in ('alpha','lambda','temperature','hfd_weight','hfd_warmup_epochs','hfd_ramp_epochs',
                      'warmup_epochs','ramp_epochs','methods','calibration_fraction','test_fraction','seeds'):
            if field in old:assert old[field]==new[field],(filename,field)


def test_completed_reader_rejects_a_twenty_epoch_result(tmp_path):
    result=tmp_path/'evaluation.json';pred=tmp_path/'predictions.npz'
    result.write_text(json.dumps({'identity':{'epochs':20}}));pred.write_bytes(b'retained prediction artifact')
    job={'id':'test','output':str(tmp_path)}
    complete={'job_id':'test','epochs':100,'evaluation_sha256':sha(result),'predictions_sha256':sha(pred)}
    (tmp_path/'budget100_completion.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError,match='non-100-epoch'):verify_completion(job)
    result.write_text(json.dumps({'identity':{'epochs':100}}));complete['evaluation_sha256']=sha(result)
    (tmp_path/'budget100_completion.json').write_text(json.dumps(complete))
    assert verify_completion(job)  # no model/projector/optimizer checkpoint required
    assert epoch_count({'big':{'selected_epoch':100},'small':{'selected_epoch':100}})==100


def test_tuning_reader_checks_actual_completed_epochs(tmp_path):
    from experiments.bidir.budget100_tuning_protocol import (
        cfg_load, install_task, load_result, make_task,
    )
    cfg=cfg_load()
    for teacher in cfg['teachers']:
        spec=make_task(cfg,'frozen',teacher,'BNCI2014004',0,666,'pair','dml_ds','eegnet')
        assert spec['teacher_training']['epochs']==100
        folder=install_task(tmp_path,spec)
        (folder/'predictions.npz').write_bytes(b'retained predictions')
        result={'task':spec,'history':[{'epoch':20}],'evaluation_scope':'validation'}
        (folder/'evaluation.json').write_text(json.dumps(result))
        with pytest.raises(ValueError,match='actually complete 100'):
            load_result(tmp_path,spec)
        result['history'][-1]['epoch']=100
        (folder/'evaluation.json').write_text(json.dumps(result))
        assert load_result(tmp_path,spec)==result
