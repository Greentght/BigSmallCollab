"""Critical invariants for validation selection and unequal effective batches."""
from fractions import Fraction
import numpy as np
import torch

from experiments.bidir.tune_fewshot_protocol import cfg_load, choose, make_task, training
from experiments.bidir.tune_fewshot_worker import nested_split, objectives, step_accumulated


def test_candidates_and_reuse():
    cfg=cfg_load();assert len(cfg['stage_a'])==len(cfg['stage_b'])==5
    for teacher in cfg['teachers']:
        a,s=training(cfg,teacher,'ifnet','A0');b,t=training(cfg,teacher,'ifnet','A4')
        assert a==b and s['batch_size']==2*t['batch_size']
        assert t['batch_size']>=cfg['micro_batch']
    # B0 is the exact same training task in both stages.
    assert make_task(cfg,'f','mirepnet','AlexMI',0,666,'pair','dml_ds','ifnet','A1') == \
           make_task(cfg,'f','mirepnet','AlexMI',0,666,'pair','dml_ds','ifnet','A1','B0')
    assert choose(['A0','A1'],{'A0':Fraction(1,3),'A1':Fraction(1,3)})=='A0'


def test_inner_split_never_evaluates_outer_test():
    cfg=cfg_load();split={'x_cal_raw':np.zeros((40,2,1000)),'y_cal':np.tile([0,1],20),
        'trial_uid_cal':np.array([f'cal_{i}' for i in range(40)]),
        'x_test_raw':np.ones((12,2,1000)),'y_test':np.tile([0,1],6),
        'trial_uid_test':np.array([f'test_{i}' for i in range(12)])}
    train,y,uids,ev,ey,euids=nested_split(split,666,cfg,'search')
    assert len(y)==32 and len(ey)==8 and not set(uids)&set(euids)
    assert set(euids)<=set(split['trial_uid_cal']) and np.all(ev==0)


def test_ds_objective_has_no_peer_gradient():
    outputs=[[torch.randn(4,2,requires_grad=True) for _ in range(3)] for _ in range(2)]
    loss=objectives(outputs,torch.tensor([0,1,0,1]),'ds_only',0.5)[0]
    grads=torch.autograd.grad(loss,outputs[0]+outputs[1],allow_unused=True)
    assert all(g is not None for g in grads[:3]) and all(g is None for g in grads[3:])


def test_accumulation_weights_tail_by_actual_samples():
    torch.manual_seed(4);x=torch.randn(10,2);y=torch.randn(10,1)
    reference=torch.nn.Linear(2,1);micro=torch.nn.Linear(2,1);micro.load_state_dict(reference.state_dict())
    ref_opt=torch.optim.SGD(reference.parameters(),lr=0.1);mic_opt=torch.optim.SGD(micro.parameters(),lr=0.1)
    for start in (0,8):
        ref_opt.zero_grad();loss=torch.nn.functional.mse_loss(reference(x[start:start+8]),y[start:start+8]);loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.0);ref_opt.step()
    mic_opt.zero_grad();count=0
    for start in (0,4,8):
        xx=x[start:start+4];loss=torch.nn.functional.mse_loss(micro(xx),y[start:start+4])*len(xx);loss.backward();count+=len(xx)
        if count==8 or start==8:step_accumulated(mic_opt,count);count=0
    assert all(torch.allclose(a,b,atol=1e-7) for a,b in zip(reference.parameters(),micro.parameters()))
