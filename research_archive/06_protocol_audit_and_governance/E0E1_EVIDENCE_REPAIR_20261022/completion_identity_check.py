"""Check approved v1.1's own digest contract and exact CSV-to-registry IDs."""
from common import *

def main():
    base=ROOT/'qoder_work/v1_1';sp=read_json(base/'SPLIT_V1_1.json');verify_sources();data=data_load();ix=np.load(base/'cal_v11_index.npy')
    if sp['state']!='FROZEN_BY_AUTHOR' or not sp['author_approval']['signed']:raise ValueError('AUTHOR_CAL_SCOPE_NOT_APPROVED')
    for name,digest in sp['outputs_sha256'].items():
        if sha256(base/name)!=digest:raise ValueError('APPROVED_V11_OUTPUT_CHANGED '+name)
    for name,digest in sp['inputs_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('APPROVED_V11_INPUT_CHANGED '+name)
    with (base/'cal_v11_windows.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not np.array_equal(np.array([int(r['sample_id']) for r in rows]),data['ids'][ix]) or not np.array_equal(np.array([r['track'] for r in rows]),data['track'][ix]):raise ValueError('V11_CSV_ID_TRACK_ROW_MISMATCH')
    if len(ix)!=1620 or int((data['groups'][ix]==2).sum())!=443:raise ValueError('V11_POPULATION_COUNTS')
    write_json(HERE/'V11_IDENTITY_REVIEW.json',dict(status='PASS_CURRENT_CAL_PARTITION_ID_CONTRACT',approved_split_sha256=sha256(base/'SPLIT_V1_1.json'),cal_rows=1620,change_rows=443,CSV_ID_track_order_exact=True,approved_input_output_SHA_exact=True,note='CAL partition only; NOT approval of whole protocol, MDE, S4, or upstream unseen exposure.'))
    print('Approvedv1.1 input/output digests and1620 CSV_ID/track rows EXACT.')

if __name__=='__main__':main()
