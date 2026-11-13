"""Bounded, resource-measured plot runner with explicit source/validator receipts."""
import json
import runpy
import subprocess
import sys
import attribution_core as c
import plot_results
import plot_export_adapter

if __name__=='__main__':
    g=c.guard();g.check(mode='startup',force=True)
    c.write_json(c.ROOT/'FIGURE_SOURCE_EXTENSION.json',dict(status='EXPLICIT_PLOT_EXTENSION',
        files={str(c.ROOT/name):c.sha(c.ROOT/name) for name in ('plot_results.py','plot_export_adapter.py','run_plots_resource.py')},
        scientific_arrays_unchanged=True))
    validator=c.PROJECT.parents[1]/'.codex/skills/nature-figure/scripts/validate_figure.py'
    # Skills live under the user profile, independent of the project folder hierarchy.
    validator=c.PROJECT.parent.parent.parent/'.codex/skills/nature-figure/scripts/validate_figure.py'
    # Explicit path anchored to the known installed skill; not a scientific data input.
    validator=__import__('pathlib').Path('<USER_HOME>/.codex/skills/nature-figure/scripts/validate_figure.py')
    r=subprocess.run([sys.executable,'-B',str(validator),str(c.ROOT/'plot_export_adapter.py'),'--json'],capture_output=True,text=True,encoding='utf-8')
    report=json.loads(r.stdout);assert report['summary']['ready'] and r.returncode==0
    c.write_json(c.ROOT/'FIGURE_SOURCE_VALIDATION.json',report)
    def export(fig,path):
        g.check();plot_export_adapter.export(fig,path)
    plot_results.save=export
    plot_results.main()
    g.check(force=True)
    c.write_json(c.ROOT/'RESOURCE_FIGURES.json',dict(policy='warning_first_remaining_v3',estimate=g.estimate,observations=g.observations))
