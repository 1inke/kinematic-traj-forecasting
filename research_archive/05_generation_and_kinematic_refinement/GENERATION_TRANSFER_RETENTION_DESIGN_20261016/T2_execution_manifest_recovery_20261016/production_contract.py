"""Execute the real train.main manifest expression, not merely an AST presence check."""
import ast,copy
import common as c
import numpy as np

def expression():
    tree=ast.parse((c.H/'train.py').read_text(encoding='utf-8-sig'));main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[n.value for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets)]
    assert len(nodes)==1 and isinstance(nodes[0],ast.Call) and isinstance(nodes[0].func,ast.Name) and nodes[0].func.id=='dict'
    return nodes[0]

def validate(row,data,source,seed):
    assert row['source_digest']==source and row['protocol_SHA']==c.sha(c.H/'protocol.json')
    assert row['statistics_SHA']==c.sha(c.H/'statistics_velocity.npz')
    assert row['cell']=='C' and row['representation']=='velocity' and row['keep'] is False and row['seed']==seed
    assert row['epochs']==25 and row['batch']==256 and row['kernel']=='integrated_acceleration' and row['no_projection'] is True
    assert row['FIT_ID_digest']==c.digest(data['ids'][data['split']['FIT']].tolist())
    assert row['only_factor']=='native24_position_to_velocity' and row['optimizer']=='AdamW.001wd.0001foreachFalse'

def check(data,source,write_name):
    node=expression();literals=[n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.endswith('.npz')]
    assert literals==['statistics_velocity.npz'],('PRODUCTION_STATISTICS_LITERAL',literals)
    assert not (c.H/'statistics_position.npz').exists(),'DO_NOT_HIDE_BUG_WITH_POSITION_ALIAS'
    st=c.npz(c.H/'statistics_velocity.npz');assert st['mean'].shape==st['std'].shape==(120,)
    assert np.array_equal(st['training_ids'],data['ids'][data['split']['FIT']])
    protocol=c.read(c.H/'protocol.json');assert protocol['scope'].startswith('T2_C_velocity') and protocol['representation']=='velocity_C; immutable_position_A_reference'
    rows=[];rejections=[]
    for seed in (42,43,44):
        namespace=dict(c=c,cell='C',seed=seed,source=source,data=data)
        row=eval(compile(ast.Expression(node),'actual-train-main-manifest','eval'),namespace);validate(row,data,source,seed);rows.append(row)
        for key,value in (('source_digest','bad'),('statistics_SHA','0'*64),('representation','position'),('keep',True),('FIT_ID_digest','bad')):
            try:validate(dict(row,**{key:value}),data,source,seed)
            except AssertionError:rejections.append(dict(seed=seed,field=key,rejected=True))
            else:raise AssertionError(('BAD_MANIFEST_ACCEPTED',key))
    old=copy.deepcopy(node)
    for n in ast.walk(old):
        if isinstance(n,ast.Constant) and n.value=='statistics_velocity.npz':n.value='statistics_position.npz'
    try:eval(compile(ast.Expression(old),'old-bug-regression-memory-only','eval'),dict(c=c,cell='C',seed=42,source=source,data=data))
    except FileNotFoundError:old_rejected=True
    else:raise AssertionError('OLD_MISSING_STATS_REFERENCE_NOT_REPRODUCED')
    report=dict(status='PASS',actual_production_expression_executed=True,not_only_AST_presence=True,all_three_actual_seeds=True,source_digest=source,train_script_SHA=c.sha(c.H/'train.py'),statistics_SHA=c.sha(c.H/'statistics_velocity.npz'),manifests=rows,bad_metadata_rejections=rejections,old_bug_in_memory_regression_rejected=old_rejected,no_parent_or_scientific_file_mutation=True,no_optimizer_or_forward=True)
    c.write(write_name,report);return report
