"""Frozen CT-NSE independent cross-dose validation. No optimizer/backward."""
from common import *
import csv,time,traceback,subprocess
from scipy import ndimage
CFG={'seed':SEED,'slices':610,'input':[1,1,512,512],'feature_shape':[32,128,128],
 'valid_feature_bounds_inclusive':[4,124],'valid_locations_per_slice':14641,
 'spatial_negative':'one deterministic uniform non-overlapping NDCT location per valid query',
 'cross_slice_negative':'one uniform NDCT location in slice (i+305)%610',
 'hard_gallery':'all 14641 full-RF locations of query slice plus slices (i+122/244/366/488)%610',
 'hard_anchors_per_slice':16,'hard_anchor_strata':'8 uniform full-RF and 8 uniform body; plus 16 visual anchors',
 'reference':'NDCT identity paired view: same deterministic input, no perturbation; upper bound, not independent sample',
 'testA':'every full-RF feature location; gallery excludes padded-RF boundary only',
 'tie_rule':'cosine within 1e-6 treated as tie, lowest gallery index wins',
 'visual_query_selection':'NDCT intensity/patch-statistic proxies only, before feature extraction/retrieval',
 'preprocessing':'raw.float32-1024 -> clip [-1024,3072] -> (HU+1024)/4096',
 'no_training':True,'no_parameter_update':True,'checkpoint':str(CHECKPOINT),'query_chunk':512}
def rows_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time();setup();access=Access()
    dump(ROOT/'effective_config.json',CFG)
    audit=json.loads((ROOT/"pairing_audit.json").read_text())
    assert audit["status"]=="PASS" and audit["slice_count"]==610
    access.active=True
    # Freeze queries and anatomy proxies before model inference and retrieval.
    rng=np.random.default_rng(SEED);xy=coords();n=len(xy)
    body=np.empty((610,n),bool);means=np.empty((610,n),np.float32)
    ldmeans=np.empty_like(means);queries=[];visual_images={};allpairs=[]
    for i in range(610):
        raw=[np.load(DATA/f'L333_{d}_{i:03d}_img.npy',allow_pickle=False) for d in ('25','target')]
        nd=raw[1].astype(np.float32)-1024
        labels,count=ndimage.label(nd>-900)
        sizes=np.bincount(labels.ravel());sizes[0]=0
        mask=ndimage.binary_fill_holes(labels==sizes.argmax())
        body[i]=mask[xy[:,0]*4,xy[:,1]*4]
        ndmean=ndimage.uniform_filter(nd,31,mode='reflect')
        means[i]=ndmean[xy[:,0]*4,xy[:,1]*4]
        lm=ndimage.uniform_filter(raw[0].astype(np.float32)-1024,31,mode='reflect')
        ldmeans[i]=lm[xy[:,0]*4,xy[:,1]*4]
        anchors=list(rng.choice(n,8,replace=False))+list(rng.choice(np.flatnonzero(body[i]),8,replace=False))
        record={'slice':i,'queries':[{'index':int(a),'yx':xy[a].tolist(),'stratum':'uniform' if j<8 else 'body'} for j,a in enumerate(anchors)],
          'hard_gallery_slices':[i]+[(i+d)%610 for d in (122,244,366,488)]}
        if i in (100,200,300,400):
            visual_images[i]=raw
            variance=np.maximum(ndimage.uniform_filter(nd*nd,31)-ndmean**2,0)
            std=np.sqrt(variance)[xy[:,0]*4,xy[:,1]*4]
            mx=ndimage.maximum_filter(nd,31)[xy[:,0]*4,xy[:,1]*4]
            smallstd=np.sqrt(np.maximum(ndimage.uniform_filter(nd*nd,7)-ndimage.uniform_filter(nd,7)**2,0))[xy[:,0]*4,xy[:,1]*4]
            tissue=body[i]&(means[i]>-160)&(means[i]<240)&(mx<400)
            s=std[tissue];q25,q75=np.percentile(s,[25,75])
            choices={'flat_soft_tissue':tissue&(std<=q25),
              'tissue_boundary':tissue&(std>=q75),
              'bone_high_contrast':body[i]&(mx>500),
              'small_scale_structure':tissue&(std>=q25)&(smallstd/np.maximum(std,1e-6)>.6)}
            for category,eligible in choices.items():
                ids=np.flatnonzero(eligible);assert len(ids)>0,(i,category)
                a=int(rng.choice(ids))
                record['queries'].append({'index':a,'yx':xy[a].tolist(),'stratum':'visual_'+category,
                    'selection_proxy':{'NDCT_meanHU':float(means[i,a]),'stdHU':float(std[a]),'maxHU':float(mx[a]),'small_stdHU':float(smallstd[a])}})
        queries.append(record)
    dump(ROOT/'query_manifest.json',{'config':CFG,'queries':queries,'selection_before_features':True,
       'visual_category_warning':'Image-statistic proxies, not radiologist annotations.'})
    model,epoch=load_model();before=state_sha(model);ck_before=sha(CHECKPOINT)
    source_hash=sha(SOURCE/'ct_nse.py')
    ld=np.empty((610,n,32),np.float32);nd=np.empty_like(ld)
    with torch.inference_mode():
        for i in range(610):
            arrays=[normalize(np.load(DATA/f'L333_{d}_{i:03d}_img.npy',allow_pickle=False)) for d in ('25','target')]
            f=model(torch.from_numpy(np.stack(arrays)[:,None]).cuda())
            assert f.shape==(2,32,128,128) and torch.isfinite(f).all()
            valid=f[:,:,4:125,4:125].flatten(2).transpose(1,2).cpu().numpy()
            ld[i]=valid[0];nd[i]=valid[1]
            if (i+1)%100==0:print('FEATURES',i+1,flush=True)
    del f,valid
    fstats={'LDCT':feature_stats(ld),'NDCT':feature_stats(nd)}
    ns=np.array(fstats['NDCT']['per_channel_std']);ls=np.array(fstats['LDCT']['per_channel_std'])
    fstats['NDCT_active_LDCT_severely_inactive_channels']=np.flatnonzero((ns>=1e-4)&(ls<.1*ns)).tolist()
    dump(ROOT/'FEATURE_STATISTICS.json',fstats)
    distributions=np.empty((2,3,610,n),np.float32)
    errors=np.empty((2,610,n),np.float32)
    top1=np.zeros((2,610,n),bool);top5=np.zeros_like(top1)
    negative_ids=np.empty((610,n),np.int32)
    cross_ids=rng.integers(0,n,size=(610,n))
    for i in range(610):
        ids=rng.integers(0,n,size=n)
        overlap=(np.abs(xy[ids]-xy)<=7).all(1)
        while overlap.any():
            ids[overlap]=rng.integers(0,n,size=int(overlap.sum()))
            overlap=(np.abs(xy[ids]-xy)<=7).all(1)
        negative_ids[i]=ids
    np.savez_compressed(ROOT/'negative_sampling_manifest.npz',spatial_ids=negative_ids,cross_ids=cross_ids)
    dump(ROOT/'negative_sampling_manifest.json',{'path':str(ROOT/'negative_sampling_manifest.npz'),
        'sha256':sha(ROOT/'negative_sampling_manifest.npz'),'cross_slice_offset':305,'seed':SEED,
        'nonoverlap_verified':bool(np.all(np.any(np.abs(xy[negative_ids]-xy[None,:,:])>7,axis=2)))})
    resultrows=[];slicerows=[];visualresults=[]
    with torch.inference_mode():
        for i,record in enumerate(queries):
            gallery=F.normalize(torch.from_numpy(nd[i]).cuda(),dim=1)
            cross=F.normalize(torch.from_numpy(nd[(i+305)%610]).cuda(),dim=1)
            inputs=(F.normalize(torch.from_numpy(ld[i]).cuda(),dim=1),gallery)
            spatial_best=np.empty((2,n),np.int32)
            for ref,q in enumerate(inputs):
                for off in range(0,n,CFG['query_chunk']):
                    stop=min(off+CFG['query_chunk'],n);ix=torch.arange(off,stop,device='cuda')
                    scores=(q[off:stop]@gallery.T).clamp(-1,1)
                    rank,best=ordinal(scores,ix)
                    pos=scores.gather(1,ix[:,None]).squeeze(1)
                    neg=torch.as_tensor(negative_ids[i,off:stop],device='cuda',dtype=torch.long)
                    nsim=scores.gather(1,neg[:,None]).squeeze(1)
                    cn=torch.as_tensor(cross_ids[i,off:stop],device='cuda',dtype=torch.long)
                    csim=(q[off:stop]*cross[cn]).sum(1).clamp(-1,1)
                    distributions[ref,0,i,off:stop]=pos.cpu().numpy()
                    distributions[ref,1,i,off:stop]=nsim.cpu().numpy()
                    distributions[ref,2,i,off:stop]=csim.cpu().numpy()
                    r=rank.cpu().numpy();b=best.cpu().numpy()
                    top1[ref,i,off:stop]=r==1;top5[ref,i,off:stop]=r<=5;spatial_best[ref,off:stop]=b
                    errors[ref,i,off:stop]=np.linalg.norm(xy[b]-xy[off:stop],axis=1)
            ids=np.array([a['index'] for a in record['queries']])
            g_slices=record['hard_gallery_slices']
            hard=F.normalize(torch.from_numpy(nd[g_slices].reshape(-1,32)).cuda(),dim=1)
            fullmeans=means[g_slices].reshape(-1)
            for ref,q in enumerate(inputs):
                scores=(q[ids]@hard.T).clamp(-1,1)
                ix=torch.as_tensor(ids,device='cuda',dtype=torch.long)
                ranks,best=ordinal(scores,ix)
                grayquery=ldmeans[i,ids] if ref==0 else means[i,ids]
                gray=torch.from_numpy(-np.abs(grayquery[:,None]-fullmeans[None,:])).cuda()
                grank,gbest=ordinal(gray,ix,tolerance=.001)
                for j,entry in enumerate(record['queries']):
                    b=int(best[j]);r=int(ranks[j]);slice_best=g_slices[b//n];yx_best=xy[b%n]
                    row={'protocol':'LDCT_to_NDCT' if ref==0 else 'NDCT_identity_to_NDCT',
                         'slice':i,'query_y':int(xy[ids[j],0]),'query_x':int(xy[ids[j],1]),
                         'stratum':entry['stratum'],'body':bool(body[i,ids[j]]),
                         'gallery_count':len(g_slices)*n,'positive_cosine':float(scores[j,ids[j]]),
                         'positive_rank':r,'top1':int(r==1),'top5':int(r<=5),'reciprocal_rank':1/r,
                         'best_slice':slice_best,'best_y':int(yx_best[0]),'best_x':int(yx_best[1]),
                         'best_cosine':float(scores[j,b]),
                         'same_slice_retrieval_top1':int(top1[ref,i,ids[j]]),
                         'same_slice_retrieval_top5':int(top5[ref,i,ids[j]]),
                         'same_slice_error_grid':float(errors[ref,i,ids[j]]),
                         'same_slice_error_pixels':float(4*errors[ref,i,ids[j]]),
                         'gray_mean_only_rank':int(grank[j]),
                         'gray_mean_only_top1':int(grank[j]==1),'gray_mean_only_top5':int(grank[j]<=5)}
                    resultrows.append(row)
                    if ref==0 and entry['stratum'].startswith('visual_'):
                        # Stable ordering for visual Top-K. Numeric aggregate uses documented tolerance rule.
                        order=torch.argsort(scores[j],descending=True,stable=True)[:5].cpu().numpy()
                        matches=[{'slice':g_slices[int(v)//n],'yx':xy[int(v)%n].tolist(),
                                  'cosine':float(scores[j,int(v)]),
                                  'feature_grid_error':float(np.linalg.norm(xy[int(v)%n]-xy[ids[j]])),
                                  'query_meanHU':float(ldmeans[i,ids[j]]),
                                  'retrieved_meanHU':float(fullmeans[int(v)])} for v in order]
                        visualresults.append({'query':row,'matches':matches,'selection_proxy':entry['selection_proxy']})
            for ref in range(2):
                for stratum,mask in (('all_full_RF',np.ones(n,bool)),('body',body[i])):
                    p,s,c=distributions[ref,:,i,mask].T if False else [distributions[ref,j,i][mask] for j in range(3)]
                    slicerows.append({'protocol':'LDCT_to_NDCT' if ref==0 else 'NDCT_identity_to_NDCT',
                        'slice':i,'stratum':stratum,'count':int(mask.sum()),
                        'positive_cosine':float(p.mean()),'spatial_negative_cosine':float(s.mean()),
                        'cross_negative_cosine':float(c.mean()),'margin_spatial':float((p-s).mean()),
                        'margin_cross':float((p-c).mean()),'positive_gt_spatial_fraction':float((p>s).mean()),
                        'positive_gt_cross_fraction':float((p>c).mean()),
                        'testA_top1':float(top1[ref,i][mask].mean()),'testA_top5':float(top5[ref,i][mask].mean()),
                        'testA_error_grid_mean':float(errors[ref,i][mask].mean())})
            if (i+1)%10==0:
                print('RETRIEVAL',i+1,'/',610,'elapsed_s',round(time.time()-start,1),flush=True)
                dump(ROOT/'STATUS.json',{'status':'RUNNING','slices_retrieved':i+1,'elapsed_seconds':time.time()-start})
    rows_csv('retrieval_results.csv',resultrows);rows_csv('per_slice_metrics.csv',slicerows)
    dump(ROOT/'VISUAL_QUERY_RESULTS.json',visualresults)
    metrics={'config':CFG,'slice_count':610,'full_RF_query_count':610*n,'valid_grid_shape':[121,121],
       'checkpoint_path':str(CHECKPOINT),'checkpoint_sha256':ck_before,'checkpoint_epoch':epoch,
       'reference_limitation':CFG['reference'],'tie_rule':CFG['tie_rule'],'feature_statistics':fstats}
    hist=[];aggregate={}
    for ref,protocol in enumerate(('LDCT_to_NDCT','NDCT_identity_to_NDCT')):
        aggregate[protocol]={}
        for stratum,mask in (('all_full_RF',np.ones_like(body)),('body',body)):
            values=[distributions[ref,j][mask] for j in range(3)]
            p,s,c=values
            d={'query_count':len(p),'positive':stats(p),'spatial_negative':stats(s),'cross_slice_negative':stats(c),
               'margin_spatial':float(np.mean(p-s)),'margin_cross_slice':float(np.mean(p-c)),
               'fraction_positive_gt_spatial_negative':float((p>s).mean()),
               'fraction_positive_gt_cross_negative':float((p>c).mean()),
               'slices_with_positive_spatial_margin':sum(x['margin_spatial']>0 for x in slicerows if x['protocol']==protocol and x['stratum']==stratum),
               'slices_with_positive_cross_margin':sum(x['margin_cross']>0 for x in slicerows if x['protocol']==protocol and x['stratum']==stratum),
               'testA':{'top1':float(top1[ref][mask].mean()),'top5':float(top5[ref][mask].mean()),
                 'mean_error_grid':float(errors[ref][mask].mean()),'median_error_grid':float(np.median(errors[ref][mask])),
                 'mean_error_pixels':float(errors[ref][mask].mean()*4),'median_error_pixels':float(np.median(errors[ref][mask])*4),
                 'gallery_count':n,'random_top1':1/n,'random_top5':5/n}}
            hr=[r for r in resultrows if r['protocol']==protocol and (stratum=='all_full_RF' or r['body'])]
            d['testB']={'query_count':len(hr),'top1':float(np.mean([r['top1'] for r in hr])),
               'top5':float(np.mean([r['top5'] for r in hr])),'MRR':float(np.mean([r['reciprocal_rank'] for r in hr])),
               'positive_rank':stats([r['positive_rank'] for r in hr]),
               'random_top1':1/(n*5),'random_top5':5/(n*5),
               'gray_mean_only_top1':float(np.mean([r['gray_mean_only_top1'] for r in hr])),
               'gray_mean_only_top5':float(np.mean([r['gray_mean_only_top5'] for r in hr]))}
            if stratum=='body':
                flat=p-s
                intensitydiff=np.abs(ldmeans-means[np.arange(610)[:,None],negative_ids])[mask]
                d['spatial_cosine_vs_gray_mean_difference_correlation']=float(np.corrcoef(s.astype(float),intensitydiff.astype(float))[0,1])
            aggregate[protocol][stratum]=d
            for kind,vals in zip(('positive','spatial_negative','cross_slice_negative'),values):
                counts,edges=np.histogram(vals,bins=100,range=(-1,1))
                for j,h in enumerate(counts):hist.append({'protocol':protocol,'stratum':stratum,'distribution':kind,'bin_low':edges[j],'bin_high':edges[j+1],'count':int(h)})
    metrics['protocols']=aggregate
    degrade={}
    for stratum in ('all_full_RF','body'):
        l=aggregate['LDCT_to_NDCT'][stratum];r=aggregate['NDCT_identity_to_NDCT'][stratum]
        degrade[stratum]={'positive_cosine_drop':r['positive']['mean']-l['positive']['mean'],
            'spatial_margin_drop':r['margin_spatial']-l['margin_spatial'],
            'cross_margin_drop':r['margin_cross_slice']-l['margin_cross_slice'],
            'testA_top1_drop_percentage_points':100*(r['testA']['top1']-l['testA']['top1']),
            'testA_top5_drop_percentage_points':100*(r['testA']['top5']-l['testA']['top5']),
            'testB_top1_drop_percentage_points':100*(r['testB']['top1']-l['testB']['top1']),
            'testB_top5_drop_percentage_points':100*(r['testB']['top5']-l['testB']['top5'])}
    metrics['cross_dose_degradation']=degrade
    rows_csv('similarity_distribution.csv',hist)
    assert state_sha(model)==before and sha(CHECKPOINT)==ck_before and sha(SOURCE/'ct_nse.py')==source_hash
    assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and not model.training
    metrics['runtime']={'elapsed_seconds':time.time()-start,'peak_allocated_VRAM_GiB':torch.cuda.max_memory_allocated()/1024**3}
    metrics['model_audit']={'eval':True,'requires_grad_all_false':True,'no_parameter_gradient':True,
        'state_sha256_before':before,'state_sha256_after':state_sha(model),
        'checkpoint_sha256_before':ck_before,'checkpoint_sha256_after':sha(CHECKPOINT),
        'source_sha256_before':source_hash,'source_sha256_after':sha(SOURCE/'ct_nse.py'),
        'optimizer_created':False,'backward_called':False}
    metrics['data_access']=access.report()
    metrics['pairing_data_access']=audit['access_audit']
    dump(ROOT/'DATA_ACCESS_AUDIT.json',access.report())
    dump(ROOT/'metrics.json',metrics)
    # Prepare structural evidence before final human visual review.
    bodymetric=aggregate['LDCT_to_NDCT']['body']
    stable=(bodymetric['margin_spatial']>0 and bodymetric['margin_cross_slice']>0
        and bodymetric['fraction_positive_gt_spatial_negative']>.5
        and bodymetric['fraction_positive_gt_cross_negative']>.5
        and bodymetric['testB']['top1']>10*bodymetric['testB']['random_top1']
        and not fstats['LDCT']['collapse'])
    metrics['numerical_decision']='CONDITIONAL PASS' if stable else 'FAIL'
    metrics['visual_review_status']='PENDING; do not claim geometric retrieval based on numbers alone'
    dump(ROOT/'metrics.json',metrics)
    from visualize import generate
    generate(metrics,visualresults,distributions,errors,body,access)
    dump(ROOT/'STATUS.json',{'status':'NUMERICAL_COMPLETE','decision':metrics['numerical_decision'],
         'visual_review_pending':True,'elapsed_seconds':time.time()-start})
    print('NUMERICAL_COMPLETE',json.dumps({'decision':metrics['numerical_decision'],'body':bodymetric,'degradation':degrade['body']}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'FAILURE.json',{'traceback':traceback.format_exc()})
        dump(ROOT/'STATUS.json',{'status':'FAILED'})
        raise
