"""Finalize only after corrected full review; retain v1 source/results/failure receipts."""
import attribution_core as c

if __name__=='__main__':
    p=c.ROOT/'finalize.py';src=p.read_text(encoding='utf-8')
    repairs={
        "'completion_review/REVIEW.json'":"'completion_review_v2/REVIEW.json'",
        "'completion_review/GROUP_SUPPLEMENT.json'":"'completion_review_v2/GROUP_SUPPLEMENT.json'",
        "'counterfactual_metrics.npz'":"'counterfactual_metrics_v2.npz'",
        'single_slot_restore.csv':'single_slot_restore_v2.csv',
        'single_slot_edit.csv':'single_slot_edit_v2.csv',
        "    c.write_json(c.ROOT/'RESULT_ANALYSIS.json',report)":
        "    report.update(counterfactual_version='v2_float64_before_observed_subtraction',precision_repair=c.json_read(c.ROOT/'PRECISION_REPAIR_V2.json'),v1_failure_preserved=True)\n    c.write_json(c.ROOT/'RESULT_ANALYSIS.json',report)",
        "    with (c.ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:":
        "    lines+=['','## 9. 本轮实现修复与保留来源','','补充审核曾因junction观察差分精度不一致失败：五端口原实现正确先float64，而新增反事实与其首版第二实现先float32差分。保留全部v1脚本、表格、原review PASS及FAILURE_REVIEW_JUNCTION_PRECISION_V1.json，不把v1审核误称覆盖正确精度。v2只重算5120集合的junction mean/p95，最大改变量1.86253e-7；路径、覆盖、quality、聚簇和其余五motion字段逐数组bitwise不变。完整第二实现从头复核v2，20组补充审核也用原float64顺序，容差仍2e-12、literal门无epsilon；以counterfactual_metrics_v2.npz、两个_v2.csv及completion_review_v2为当前结果。近邻不足5条的NaN填充均值与收据字段名亦有显式review adapter，旧源码保留，未改原科学。']\n    with (c.ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:"
    }
    for old,new in repairs.items():assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    c.write_json(c.ROOT/'FINALIZATION_V2_EXTENSION.json',dict(files={str(p):c.sha(p),str(c.ROOT/'finalize_v2_adapter.py'):c.sha(c.ROOT/'finalize_v2_adapter.py')},
        status='EXPLICIT_CURRENT_REVIEW_AND_CORRECTED_OUTPUT_BINDING',replacements=repairs,old_source_preserved=True))
    ns={'__name__':'finalize_v2_namespace','__file__':str(p)}
    exec(compile(src,str(p),'exec'),ns);ns['main']()
