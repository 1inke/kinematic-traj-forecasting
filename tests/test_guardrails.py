import copy
import pytest
from f16.guardrails import measure_guardrails

def test_G3_must_not_use_minADE():
    row=dict(overall={'coverage':.7},change={'motion_pass':True,'Q_winner_ADE':.2,'Q_winner_FDE':.3,'ADE':.1,'FDE':.1,'lost_rate_native_good':.04})
    t=copy.deepcopy(row);t['change']['Q_winner_ADE']=.21
    guards,_=measure_guardrails([t],[row],[0,.1],total_parameters=250000,seed_policy='all_seeds')
    assert guards['G3'] is False and all(v for k,v in guards.items() if k!='G3')

def test_unknown_lost_denominator_does_not_pass():
    row=dict(overall={'coverage':.7},change={'motion_pass':True,'Q_winner_ADE':.2,'Q_winner_FDE':.3,'lost_rate_native_good':None})
    guards,_=measure_guardrails([row],[row],[0,.1],total_parameters=250000,seed_policy='all_seeds')
    assert guards['G4'] is None

def test_budget_is_whole_model_not_just_deltahead():
    row=dict(overall={'coverage':.7},change={'motion_pass':True,'Q_winner_ADE':.2,'Q_winner_FDE':.3,'lost_rate_native_good':0})
    guards,_=measure_guardrails([row],[row],[0,.1],total_parameters=355048,seed_policy='all_seeds')
    assert guards['G5'] is False
    with pytest.raises(ValueError):measure_guardrails([row],[row],[0,.1],total_parameters=250000,seed_policy='unregistered')
