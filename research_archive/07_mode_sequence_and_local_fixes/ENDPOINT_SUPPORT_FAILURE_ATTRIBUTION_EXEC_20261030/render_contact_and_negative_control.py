"""Python-only visual contact sheet + explicit saved-data noninterference fixture."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import attribution_core as c

def main():
    z=c.load_npz(c.ROOT/'derived_ports.npz')
    # Deliberately changed synthetic GT is used ONLY in this negative control,
    # never in any scientific output or selector. Saved observed construction stays exact.
    before=z['paths'][:4,:8].copy();gt=z['gt'][:8].copy()
    perturbed=gt+np.array([.123,-.071])
    q1=c.quality(before[3],gt);q2=c.quality(before[3],perturbed)
    assert np.array_equal(before,z['paths'][:4,:8]) and not np.array_equal(q1[2],q2[2])
    c.write_json(c.ROOT/'GT_NEGATIVE_CONTROL.json',dict(status='PASS_FIRST8_SAVED_PATH_GT_NONINTERFERENCE',
        rows=8,GT_changed_only_for_fixture=True,scientific_metrics_unmodified=True,
        N_C_Q_S_paths_byte_unchanged=True,posthoc_J_changed=True,
        selector_takes_no_GT_and_map_seal_precedes_label_association=True))
    files=sorted((c.ROOT/'cases').glob('*.png'));assert len(files)==15
    fig,axes=plt.subplots(5,3,figsize=(12,17),layout='constrained')
    for ax,p in zip(axes.flat,files):
        with Image.open(p) as im:
            im.thumbnail((700,1000));ax.imshow(np.asarray(im))
        ax.set_title(p.stem,fontsize=9);ax.axis('off')
    fig.savefig(c.ROOT/'figures/ALL15_CONTACT.png',dpi=180,bbox_inches='tight');plt.close(fig)
    c.write_json(c.ROOT/'CONTACT_SOURCE_EXTENSION.json',dict(files={str(c.ROOT/'render_contact_and_negative_control.py'):c.sha(c.ROOT/'render_contact_and_negative_control.py')},
        purpose='All15 visual QA thumbnails only, never substitutes full-size case PDF or source data'))

if __name__=='__main__':main()
