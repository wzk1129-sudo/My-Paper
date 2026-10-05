"""Diagnostic PNGs and a report; all private images remain remote by default."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def generate(metrics,visualresults,distributions,errors,body,access):
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,ref,label in zip(axes,(0,1),('LDCT -> NDCT','NDCT identity -> NDCT')):
        for j,name in enumerate(('positive','spatial negative','cross-slice negative')):
            vals=distributions[ref,j][body]
            ax.hist(vals,bins=80,range=(0,1),density=True,histtype='step',label=name)
        ax.set_title(label+' (body)');ax.set_xlabel('Cosine');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'visuals/cosine_distribution.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for ref,label in enumerate(('LDCT -> NDCT','NDCT identity -> NDCT')):
        ax.hist(errors[ref][body],bins=70,histtype='step',density=True,label=label)
    ax.set_xlabel('Same-slice Top-1 spatial error, feature-grid units (x4 = pixels)')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'visuals/spatial_error_distribution.png',dpi=160);plt.close(fig)
    image_cache={}
    def patch(i,dose,yx):
        key=(i,dose)
        if key not in image_cache:
            image_cache[key]=np.load(DATA/f'L333_{dose}_{i:03d}_img.npy',allow_pickle=False).astype(np.float32)-1024
        y,x=np.array(yx)*4
        return image_cache[key][y-15:y+16,x-15:x+16]
    fig,axes=plt.subplots(len(visualresults),7,figsize=(17.5,2.1*len(visualresults)),squeeze=False)
    metadata=[]
    for row,item in enumerate(visualresults):
        q=item['query'];yx=[q['query_y'],q['query_x']];i=q['slice']
        patches=[patch(i,'25',yx),patch(i,'target',yx)]+[patch(m['slice'],'target',m['yx']) for m in item['matches']]
        reference=patches[1];quant=[]
        for col,(ax,p) in enumerate(zip(axes[row],patches)):
            ax.imshow(p,cmap='gray',vmin=-160,vmax=240,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:title=f"LDCT S{i} {yx}\n{q['stratum'].replace('visual_','')}"
            elif col==1:title=f"Paired NDCT S{i} {yx}\nmean {p.mean():.1f} std {p.std():.1f} HU"
            else:
                m=item['matches'][col-2]
                err=(f"{4*m['feature_grid_error']:.1f}px" if m['slice']==i else 'other slice')
                a=(p-p.mean()).ravel();b=(reference-reference.mean()).ravel()
                corr=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
                title=f"Top{col-1} S{m['slice']} {m['yx']}\ncos {m['cosine']:.4f}; {err}\nmean {p.mean():.1f}; demean corr {corr:.2f}"
                quant.append({'rank':col-1,'meanHU':float(p.mean()),'stdHU':float(p.std()),
                              'demeaned_patch_correlation_to_paired_NDCT':corr,'match':m})
            ax.set_title(title,fontsize=8)
        metadata.append({'query':q,'selection_proxy':item['selection_proxy'],
            'paired_NDCT_meanHU':float(reference.mean()),'paired_NDCT_stdHU':float(reference.std()),'matches':quant})
    fig.suptitle('Frozen CT-NSE cross-dose retrieval | CT display [-160,240] HU | patches 31x31\nQueries selected before retrieval from image-statistic proxies; colors/intensity are not semantic labels.',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.98])
    fig.savefig(ROOT/'visuals/retrieval_contact_sheet.png',dpi=140);plt.close(fig)
    dump(ROOT/'visuals/RETRIEVAL_PATCH_DIAGNOSTICS.json',metadata)
    # The written report describes numerical evidence and awaits explicit human visual assessment.
    protocols=metrics['protocols'];lines=['# CT_NSE_P_V1 cross-dose independent diagnostic','',
      'Numerical evaluation complete. Final visual assessment pending; preliminary decision: '+metrics['numerical_decision']+'.','',
      '## Isolation and pairing','',
      'Frozen best_val_loss checkpoint: '+str(CHECKPOINT), 'SHA256: '+metrics['checkpoint_sha256'],
      'Model eval(), requires_grad=False, inference_mode; no optimizer, backward, training, checkpoint/source changes.',
      '610 strictly paired L333 slices (zero-based 0..609), exact original DICOM/NPY correspondence and physical geometry verified.',
      'Preprocessing: raw.float32-1024 -> clip [-1024,3072] HU -> (HU+1024)/4096. Full 512x512 inputs; no registration, extra clip or resize.',
      'Full RF grid [4,124] inclusive; 14641 valid locations per slice, 8931010 total.',
      'All-location and body-only results are both reported. Body diagnostic mask: HU>-900, largest connected component, hole filling; it does not alter input.',
      'L333 supplied validation/checkpoint selection in original NSE training. This is a new cross-dose diagnostic, NOT an unseen-patient generalization test.','',
      '## Sampling and reference','',
      CFG_text(metrics),'',
      'NDCT identity reference is an upper bound using identical deterministic paired views. It is not a noise or augmentation robustness test.',
      'One spatial negative per location sampled uniformly with RF overlap excluded (abs dx<=7 and abs dy<=7); one cross negative from (i+305)%610, uniformly sampled location.',
      'Test A covers every valid location and the entire valid same-slice gallery.',
      'Test B uses 16 fixed anchors per slice (8 uniform, 8 body) plus 16 visualization queries, against full galleries of five slices.',
      'Cross gallery offsets 122/244/366/488; minimum cyclic/absolute separation 122. Query/negative manifests were fixed before retrieval.',
      'Ties within 1e-6 are resolved by lower gallery index; exact top1/5 are coordinate identification, not merely anatomical equivalence.','',
      '## Main metrics','',
      '| Protocol | Stratum | Positive mean | Spatial neg mean | Cross neg mean | Spatial margin | Cross margin | P>spatial | P>cross | A Top1 | A Top5 | B Top1 | B Top5 | B MRR |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for protocol,groups in protocols.items():
        for stratum,d in groups.items():
            lines.append(f"| {protocol} | {stratum} | {d['positive']['mean']:.6f} | {d['spatial_negative']['mean']:.6f} | {d['cross_slice_negative']['mean']:.6f} | {d['margin_spatial']:.6f} | {d['margin_cross_slice']:.6f} | {d['fraction_positive_gt_spatial_negative']:.6f} | {d['fraction_positive_gt_cross_negative']:.6f} | {d['testA']['top1']:.6f} | {d['testA']['top5']:.6f} | {d['testB']['top1']:.6f} | {d['testB']['top5']:.6f} | {d['testB']['MRR']:.6f} |")
    lines+=['','Full mean/median/std/P5/P25/P75/P95, Test A error in grid and pixel units, Test B positive rank distribution, random chance levels, mean-HU-only baseline and per-channel stats are in metrics.json.',
       'similarity_distribution.csv contains histogram bin counts; per_slice_metrics.csv has one row per slice/protocol/stratum.',
       'retrieval_results.csv contains the exact fixed-query ranks, coordinates, similarities and spatial errors.','',
       '## Cross-dose degradation','',json.dumps(metrics['cross_dose_degradation'],indent=2),'',
       '## Feature integrity','',json.dumps(metrics['feature_statistics'],indent=2),'',
       '## Q1-Q4 and scope','',
       'Q1/Q2: Use the body/all-location cosine margins, per-slice signs and positive>negative fractions above to assess same-position alignment and wrong-location separation.',
       'Q3: Drops from identity NDCT reference are reported explicitly above. The identity reference is deliberately an upper bound; duplicate/background features can still tie.',
       'Q4: Numerical decision is preliminary pending manual inspection of the 16-query contact sheet. A downstream frozen-NSE RDDM structural-loss trial is not launched.',
       'High cosine or retrieval alone cannot prove feature independence from HU intensity. Mean-HU-only retrieval baseline and demeaned patch correlations help contextualize, but are not anatomical labels.',
       '## Provenance and access audit','',json.dumps(metrics['model_audit'],indent=2),json.dumps(metrics['data_access'],indent=2),
       'Runtime: '+json.dumps(metrics['runtime']), '',
       'Outputs: visuals/cosine_distribution.png, visuals/spatial_error_distribution.png, visuals/retrieval_contact_sheet.png.',
       'No L506, L067/L143 or external memory was read; no training was run; no commit/push was performed.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def CFG_text(metrics):
    return json.dumps(metrics['config'],indent=2)
