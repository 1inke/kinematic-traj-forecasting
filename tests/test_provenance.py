import pytest
from f16.io import sha256,write_json
from f16.provenance import verify_completed

def test_bad_completed_output_source_rejected(tmp_path):
    p=tmp_path/'output.json';write_json(p,{'result':1});m=tmp_path/'done.json';contract={'source':'s','IDs':'i','protocol':'p'}
    write_json(m,dict(contract=contract,files={'output.json':sha256(p)}));assert verify_completed(tmp_path,m,contract)
    with pytest.raises(ValueError):verify_completed(tmp_path,m,{**contract,'source':'bad'})
    write_json(p,{'result':2})
    with pytest.raises(ValueError):verify_completed(tmp_path,m,contract)
