"""Reporting only: never stops training or extends its budget."""

def plateau(history):
    rows={r['step']:r for r in history if r['validation_mode']=='FAST' and r['slices']==64}
    if 6000 not in rows or 8000 not in rows:
        return {'status':'INSUFFICIENT_EVIDENCE','delta_psnr_6000_to_8000':None,'automatic_early_stop':False}
    delta=rows[8000]['psnr']-rows[6000]['psnr']
    return {'status':'PLATEAU' if delta<0.01 else 'NOT_PLATEAU',
            'delta_psnr_6000_to_8000':delta,'automatic_early_stop':False}

def promotion(candidate,baseline):
    """Engineering screening only; callers verify identical fixed protocol first.

    Any SSIM/RMSE/MAE reverse direction is conservatively routed to review because
    'material degradation' has no user-specified numerical tolerance. No p-value
    or statistical-significance claim is implied.
    """
    import math
    c={r['step']:r for r in candidate if r['validation_mode']=='FAST' and r['slices']==64}
    b={r['step']:r for r in baseline if r['validation_mode']=='FAST' and r['slices']==64}
    if not all(s in c and s in b for s in (6000,8000)):
        return {'status':'INSUFFICIENT_EVIDENCE'}
    delta={str(s):c[s]['psnr']-b[s]['psnr'] for s in (6000,8000)}
    threshold=delta['8000']>=0.05 or math.isclose(delta['8000'],0.05,rel_tol=0,abs_tol=1e-12)
    passed=threshold and all(x>0 for x in delta.values())
    regressions=[];missing=[]
    for step in (6000,8000):
        for metric,higher in [('ssim',True),('rmse_hu',False),('mae_hu',False)]:
            if metric not in c[step] or metric not in b[step]:missing.append([step,metric]);continue
            difference=c[step][metric]-b[step][metric]
            if difference<0 if higher else difference>0:
                regressions.append({'step':step,'metric':metric,'method_minus_baseline':difference})
    status='KEEP_IN_DEVELOPMENT'
    if passed:status='REVIEW_OTHER_METRICS' if regressions or missing else 'PROMOTE_TO_L333_FULL'
    return {'status':status,'delta_psnr':delta,'metric_regressions':regressions,'missing_metrics':missing,
            'heuristic_only_not_statistical_significance':True,'same_checkpoint_full_L333_review_required':True,
            'positive_full_L333_gain_required_before_formal_training':True,
            'final_paper_requires_fresh_8_patient_training':True}
