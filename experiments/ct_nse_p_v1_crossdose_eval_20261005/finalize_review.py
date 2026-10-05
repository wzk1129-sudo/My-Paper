"""Finalize diagnostic after explicit human visual review; no model inference."""
from pathlib import Path
import json,csv,hashlib,math,datetime
ROOT=Path(__file__).resolve().parent
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((ROOT/'metrics.json').read_text())
v=json.loads((ROOT/'visuals/RETRIEVAL_PATCH_DIAGNOSTICS.json').read_text())
review={'status':'COMPLETE','review_type':'visual inspection of 16-query contact sheet; qualitative, not a radiologist annotation',
 'contact_sheet_sha256':sha(ROOT/'visuals/retrieval_contact_sheet.png'),
 'exact_top1_among_visual_queries':sum(x['matches'][0]['match']['slice']==x['query']['slice'] and x['matches'][0]['match']['yx']==[x['query']['query_y'],x['query']['query_x']] for x in v),
 'visual_query_count':16,
 'observations':[
  'Several high-contrast contours and tissue boundaries retrieve the exact patch or similar coarse geometry across slices.',
  'For S100 (32,48) and S200 (35,92), non-coordinate Top1 still has similar boundary layout; demeaned patch correlations are 0.877 and 0.825.',
  'Flat/proxy-flat queries S100 (46,50), S200 (96,54) and S300 (89,29) have Top1 demeaned correlations 0.060, -0.082 and 0.062. High NSE cosine does not establish fine texture identity.',
  'S100 tissue-boundary proxy (72,73) also has an incorrect Top1 with weak patch correlation; category labels are image-statistic proxies, not expert anatomical labels.',
  'S300 small-scale proxy (80,46) retrieves a different slice with only moderate geometric agreement (demeaned correlation 0.399).',
  'NSE is not a guaranteed fine-grained anatomical identity matcher. Mean-HU-only hard retrieval is much weaker, so the features are not explained by scalar mean HU alone, but density/coarse-layout dependence remains.'],
 'final_decision':'CONDITIONAL PASS',
 'reason':'Cross-dose same-position consistency and coarse discrimination are strong without collapse; exact retrieval degrades substantially and some weak-texture/boundary queries are ambiguous.'}
write(ROOT/'VISUAL_REVIEW.json',review)
m['visual_review_status']='COMPLETE'
m['visual_review']=review
m['final_decision']='CONDITIONAL PASS'
m['decision_reason']=review['reason']
ndkeys={x['query']['slice'] for x in v}|{r['match']['slice'] for x in v for r in x['matches']}
ldkeys={x['query']['slice'] for x in v}
m['data_access']['visualization_reader_cache_unique_NPY']={'LDCT':len(ldkeys),'NDCT':len(ndkeys)}
m['data_access']['visualization_count_note']='Recorded event counts in this object precede visualization. Additional cache reads are reconstructed from saved query/match metadata and the reader code, one np.load per unique (slice,dose). The same allowlist guard stayed active through visualization.'
m['local_preview']={'explicitly_authorized':True,'only_file':'retrieval_contact_sheet.png',
 'path':r'D:\ShuJuSHEN\experiment_records\ct_nse_p_v1_crossdose_eval_20261005\retrieval_contact_sheet.png',
 'sha256':review['contact_sheet_sha256'],'medical_arrays_features_copied':False}
write(ROOT/'metrics.json',m);write(ROOT/'DATA_ACCESS_AUDIT.json',{'inference_phase':m['data_access'],'pairing_phase':m['pairing_data_access']})
pair=json.loads((ROOT/'pairing_audit.json').read_text())
assert pair['status']=='PASS' and pair['slice_count']==610 and len(pair['pairs'])==610
assert all(x['strict_pairing'] and all(y['DICOM_NPY_exact'] for y in x['files'].values()) for x in pair['pairs'])
for access in (m['data_access'],m['pairing_data_access']):
 assert access['L506_access']==access['L067_L143_access']==access['external_memory_access']==access['blocked_attempts']==0
a=m['model_audit']
assert a['state_sha256_before']==a['state_sha256_after'] and a['checkpoint_sha256_before']==a['checkpoint_sha256_after']==sha(m['checkpoint_path'])
manifest=json.loads((ROOT/'SOURCE_MANIFEST_RETRY1.json').read_text())
assert all(sha(ROOT.parent.parent/p)==h for p,h in manifest.items())
sr=list(csv.DictReader((ROOT/'per_slice_metrics.csv').open()))
rr=list(csv.DictReader((ROOT/'retrieval_results.csv').open()))
assert len(sr)==2440 and len(rr)==19552 and all(1<=int(r['positive_rank'])<=73205 for r in rr)
assert all(int(r['top1'])==int(int(r['positive_rank'])==1) and int(r['top5'])==int(int(r['positive_rank'])<=5) for r in rr)
for protocol,groups in m['protocols'].items():
 for stratum,d in groups.items():
  rows=[r for r in sr if r['protocol']==protocol and r['stratum']==stratum]
  assert len(rows)==610
  assert sum(int(r['count']) for r in rows)==d['query_count']
  for col in ('positive_cosine','spatial_negative_cosine','cross_negative_cosine','margin_spatial','margin_cross','testA_top1','testA_top5'):
   mapped={'positive_cosine':d['positive']['mean'],'spatial_negative_cosine':d['spatial_negative']['mean'],
    'cross_negative_cosine':d['cross_slice_negative']['mean'],'margin_spatial':d['margin_spatial'],
    'margin_cross':d['margin_cross_slice'],'testA_top1':d['testA']['top1'],'testA_top5':d['testA']['top5']}[col]
   computed=sum(int(r['count'])*float(r[col]) for r in rows)/d['query_count']
   assert abs(computed-mapped)<2e-6,(col,computed,mapped)
 assert sum(float(r['margin_spatial'])>0 and float(r['margin_cross'])>0 for r in sr if r['protocol']==protocol and r['stratum']=='body')==610
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
finite(m)
f=m['feature_statistics'];assert not f['LDCT']['collapse'] and not f['NDCT']['collapse']
assert f['LDCT']['nan_count']==f['LDCT']['inf_count']==f['NDCT']['nan_count']==f['NDCT']['inf_count']==0
integrity={'status':'PASS','slice_count':610,'full_RF_query_count':8931010,'body_query_count':4492418,
 'hard_queries':9776,'hard_result_rows_both_protocols':len(rr),'per_slice_rows':len(sr),
 'all_pairing_physical_and_DICOM_NPY_checks':True,'source_config_checkpoint_unchanged':True,
 'model_frozen_and_no_parameter_update':True,'per_slice_weighted_aggregate_reconciles':True,
 'rank_flags_reconcile':True,'all_reported_numbers_finite':True,
 'feature_NaN_Inf':0,'L506_access':0,'L067_L143_access':0,'external_memory_access':0,
 'decision':'CONDITIONAL PASS','medical_copy':'one explicitly authorized contact sheet PNG; no arrays/features',
 'completion_time':datetime.datetime.now().astimezone().isoformat()}
write(ROOT/'FINAL_INTEGRITY_AUDIT.json',integrity)
d=m['protocols']['LDCT_to_NDCT']['body'];r=m['protocols']['NDCT_identity_to_NDCT']['body'];de=m['cross_dose_degradation']['body']
lines=['','## Final answers after visual review','',
 '**Final decision: CONDITIONAL PASS.** '+review['reason'],'',
 'Q1: Yes, same-position LDCT/NDCT features are close. Body positive mean cosine %.6f, median %.6f, P5 %.6f.'%(d['positive']['mean'],d['positive']['median'],d['positive']['p5']),
 'Q2: Yes, random nonoverlapping spatial and distant-slice negatives are clearly separated: margins %.6f and %.6f; positive>negative %.4f%% and %.4f%%. All 610 slices have both body margins positive.'%(d['margin_spatial'],d['margin_cross_slice'],100*d['fraction_positive_gt_spatial_negative'],100*d['fraction_positive_gt_cross_negative']),
 'Q3: Against deterministic identity NDCT upper bound, positive cosine drops %.6f and spatial/cross margins drop %.6f/%.6f. Exact-coordinate Test A Top1/Top5 drop %.3f/%.3f percentage points; Test B drops %.3f/%.3f points. This is a substantial rank degradation despite high positive cosine; it is not complete loss of discrimination.'%(de['positive_cosine_drop'],de['spatial_margin_drop'],de['cross_margin_drop'],de['testA_top1_drop_percentage_points'],de['testA_top5_drop_percentage_points'],de['testB_top1_drop_percentage_points'],de['testB_top5_drop_percentage_points']),
 'Q4: Evidence supports considering a carefully bounded frozen-NSE structural-loss pilot, with coarse-structure and intensity-sensitivity caveats. It does not establish downstream restoration improvement or justify treating the features as noise-independent anatomical ground truth. No next-stage training is started.','',
 'Body hard retrieval mean-HU-only Top1 %.4f%% versus NSE %.4f%%; scalar mean HU alone is insufficient to explain NSE performance. However spatial-negative cosine versus HU-mean difference has correlation %.4f, indicating strong intensity dependence that the random-negative separation alone cannot remove.'%(100*d['testB']['gray_mean_only_top1'],100*d['testB']['top1'],d['spatial_cosine_vs_gray_mean_difference_correlation']),
 'Body Test A mean spatial error %.4f grid / %.4f pixels, median %.4f grid / %.4f pixels. Large-error tails remain.'%(d['testA']['mean_error_grid'],d['testA']['mean_error_pixels'],d['testA']['median_error_grid'],d['testA']['median_error_pixels']),
 'LDCT global std %.6f versus NDCT %.6f; near-zero channels 9 in both. LDCT zero channels 9 versus NDCT 8; the extra NDCT channel is already near-zero (std 1.91e-6), not an active channel lost cross-dose. No NDCT-active channel becomes severely inactive on LDCT.'%(f['LDCT']['global_std'],f['NDCT']['global_std']),
 '## Visual observations','']+['- '+x for x in review['observations']]+['',
 '7/16 visualization queries retrieve the exact paired coordinate as Top1. This small selected proxy-category panel is a qualitative check, not a population accuracy estimate.',
 'The contact sheet was explicitly authorized for local preview; its SHA256 is '+review['contact_sheet_sha256']+'. No other PNG/array/features were copied.',
 '## Completion and independent file verification','',json.dumps(integrity,indent=2),
 'All outputs remain in the new directory; original training artifacts and checkpoint were unchanged. No training, install, commit or push. STOP.']
with (ROOT/'REPORT.md').open('a') as out:out.write('\n'.join(lines)+'\n')
write(ROOT/'STATUS.json',{'status':'COMPLETE','final_decision':'CONDITIONAL PASS','visual_review_complete':True,'final_integrity':'PASS','completion_time':integrity['completion_time']})
outputs={str(x.relative_to(ROOT)):{'bytes':x.stat().st_size,'sha256':sha(x)} for x in ROOT.rglob('*') if x.is_file() and '__pycache__' not in str(x) and x.name!='OUTPUT_MANIFEST.json'}
write(ROOT/'OUTPUT_MANIFEST.json',outputs)
print(json.dumps(integrity,indent=2))
