"""Collect matched evidence; run only the authorized promotion and inference diagnostics."""
import csv,json,math,gc
from pathlib import Path
import numpy as np
import torch
from model import build_model,load_state_without_med
from provenance import ROOT,write_json
from fast_dev.protocol import CONFIG,check_config
from mayo2016_smallset_v1.runtime import seed_everything,METRICS
from context_data import ContextPairs,ContextDataGuard
from evaluate import validate
from train import ARMS,STEPS
def csv_write(path,rows):
    assert rows
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def json_read(path):return json.loads(path.read_text())
def run_paths():
    mapping=ROOT/"RUN_PATHS.json"
    if mapping.exists():
        paths={a:Path(v).resolve() for a,v in json_read(mapping).items()}
        assert set(paths)==set(ARMS) and all(ROOT/"runs" in p.parents for p in paths.values())
        return paths
    return {a:ROOT/"runs"/a for a in ARMS}
def eval_checkpoint(arm,mode,name,condition="NORMAL"):
    cfg=json_read(CONFIG);check_config(cfg)
    seed_everything(20260829);device=torch.device("cuda:0")
    output=run_paths()[arm]
    ck=torch.load(output/"checkpoints/step_8000.pt",map_location="cpu",weights_only=False)
    assert ck["optimizer_step"]==ck["ema_updates"]==8000
    model=build_model(arm==ARMS[1]).to(device)
    load_state_without_med(model,ck["ema_state_dict"])
    del ck;gc.collect()
    with ContextDataGuard(cfg,mode,allow_training=False) as guard:
        data=ContextPairs(cfg,"validation",mode);data.bind_guard(guard)
        result,activity=validate(model,data,cfg,device,ROOT/(name+"_PER_SLICE.json"),condition)
        access=guard.report()
    assert access["L506_opens"]==access["external_memory_opens"]==access["neighbor_target_opens"]==access["blocked_attempts"]==0
    write_json(ROOT/(name+".json"),{"metrics":result,"activity":activity,"data_access":access})
    del model;gc.collect();torch.cuda.empty_cache()
    print("POST_EVALUATION "+json.dumps({"name":name,**result}),flush=True)
    return result,access
def main():
    cfg=json_read(CONFIG);check_config(cfg)
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    paths=run_paths()
    final={a:json_read(p/"FINAL_STATE.json") for a,p in paths.items()}
    assert all(x["optimizer_step"]==x["ema_updates"]==8000 and x["parameters_finite"] for x in final.values())
    initial={a:json_read(p/"INITIAL_STATE.json") for a,p in paths.items()}
    for key in ("shared_initial_parameters_sha256","initial_rng_sha256","train_manifest_sha256","validation_manifest_sha256","batch_config"):
        assert initial[ARMS[0]][key]==initial[ARMS[1]][key],key
    logs={}
    for a,p in paths.items():
        with (p/"train.jsonl").open() as f:logs[a]=[json.loads(x) for x in f]
        assert len(logs[a])==8000
        assert [r["optimizer_step"] for r in logs[a]]==list(range(1,8001))
        assert all(math.isfinite(r["attention_gradient_norm"]) and r["attention_gradient_norm"]>0 for r in logs[a])
        assert all(r["parameters_finite"] is not False for r in logs[a])
    first={a:logs[a][0] for a in ARMS}
    for key in ("samples","timesteps","noise_sha256","microbatch_noise_sha256","pre_step_rng_sha256","post_step_rng_sha256"):
        assert first[ARMS[0]][key]==first[ARMS[1]][key],key
    assert all(x["train_stream_sha256"]==y["train_stream_sha256"] for x,y in zip(logs[ARMS[0]],logs[ARMS[1]]))
    history={}
    for a,p in paths.items():
        with (p/"validation_metrics.csv").open() as f:history[a]={int(r["step"]):r for r in csv.DictReader(f)}
        assert tuple(history[a])==STEPS
    table=[];per_slice=[]
    for step in STEPS:
        c,n=history[ARMS[0]][step],history[ARMS[1]][step]
        row={"step":step}
        for m in METRICS:
            row["control_"+m]=float(c[m]);row["candidate_"+m]=float(n[m]);row["delta_"+m]=float(n[m])-float(c[m])
        assert c["train_stream_sha256"]==n["train_stream_sha256"]
        row["train_stream_sha256"]=c["train_stream_sha256"];table.append(row)
        slices={a:json_read(paths[a]/("validation_step_%d.json"%step)) for a in ARMS}
        assert len(slices[ARMS[0]])==len(slices[ARMS[1]])==64
        for x,y in zip(slices[ARMS[0]],slices[ARMS[1]]):
            assert x["slice_index"]==y["slice_index"] and x["sampling_seed"]==y["sampling_seed"]
            r={"step":step,"slice_index":x["slice_index"]}
            for m in METRICS:r["control_"+m]=x[m];r["candidate_"+m]=y[m];r["delta_"+m]=y[m]-x[m]
            per_slice.append(r)
    assert abs(table[0]["delta_psnr"])<=1e-9 and all(abs(table[0]["delta_"+m])<=1e-9 for m in METRICS)
    csv_write(ROOT/"VALIDATION_HISTORY.csv",table);csv_write(ROOT/"PER_SLICE_DELTA.csv",per_slice)
    activity=[]
    for step in STEPS:
        d=json_read(paths[ARMS[1]]/("condition_activity_step_%d.json"%step))
        assert len(d)==9
        for stage,values in d.items():activity.append({"step":step,"stage":stage,**values,"native_attention_gradient_norm":float(history[ARMS[1]][step]["attention_gradient_norm"])})
    csv_write(ROOT/"CONDITION_ACTIVITY.csv",activity)
    diagnostics=[{"arm":a,**{k:r[k] for k in ("optimizer_step","train_loss","gradient_norm_before_clip","attention_gradient_norm","parameters_finite","seconds_per_step","peak_vram_bytes","train_stream_sha256")}} for a in ARMS for r in logs[a]]
    csv_write(ROOT/"TRAINING_DIAGNOSTICS.csv",diagnostics)
    audit={"arms":{a:json_read(p/"DATA_ACCESS_AUDIT.json") for a,p in paths.items()},
        "matched_random_stream":{"all_8000_step_hashes_equal":True,"shared_initial_parameters_equal":True,"first_batch":{k:first[ARMS[0]][k] for k in ("samples","timesteps","noise_sha256","microbatch_noise_sha256","pre_step_rng_sha256","post_step_rng_sha256")},"final_stream_sha256":final[ARMS[0]]["train_stream_sha256"]},
        "readiness":json_read(ROOT/"READINESS.json")["data_access_audit"]}
    assert final[ARMS[1]]["meddino_weights_unchanged"] is True
    for v in audit["arms"].values():assert v["L506_opens"]==v["external_memory_opens"]==v["neighbor_target_opens"]==v["blocked_attempts"]==0
    last=table[-1];better=sum(r["delta_psnr"]>0 for r in per_slice if r["step"]==8000)
    promotion=last["delta_psnr"]>=.03 and better/64>=.6
    full=None;inference=None
    if promotion:
        full={};full_audit={}
        for a in ARMS:
            metrics,access=eval_checkpoint(a,"FULL","FULL_L333_"+a)
            assert metrics["slices"]==610
            full[a]=metrics;full_audit[a]=access
        full["delta"]={m:full[ARMS[1]][m]-full[ARMS[0]][m] for m in METRICS}
        audit["full_l333"]=full_audit
    if last["delta_psnr"]>0:
        inference={};inf_audit={}
        for mode in ("NORMAL","CONDITION_OFF","SHUFFLED_MEDDINO"):
            metrics,access=eval_checkpoint(ARMS[1],"FAST","DIAGNOSTIC_"+mode,mode)
            inference[mode]=metrics;inf_audit[mode]=access
        assert abs(inference["NORMAL"]["psnr"]-last["candidate_psnr"])<=1e-8
        audit["inference_diagnostics"]=inf_audit
    write_json(ROOT/"DATA_ACCESS_AUDIT.json",audit)
    source=json_read(ROOT/"SOURCE_PROVENANCE.json")
    parent={m:float(np.mean([r[m] for r in source["parent_8k_metrics"]])) for m in METRICS}
    summary={"status":"COMPLETE","arms":list(ARMS),"continuation_steps":8000,"parent_2d_8k":parent,
        "batch_config":initial[ARMS[0]]["batch_config"],"control_8k":{m:last["control_"+m] for m in METRICS},
        "candidate_8k":{m:last["candidate_"+m] for m in METRICS},"delta_8k":{m:last["delta_"+m] for m in METRICS},
        "improved_psnr_slices":better,"total_fast_slices":64,"improved_psnr_fraction":better/64,
        "promotion_gate_passed":promotion,"full_l333_executed":promotion,"full_l333":full,
        "inference_diagnostics":inference,"meddino_frozen":True,"meddino_weights_unchanged":True,
        "native_attention_forward_backward_verified":True,"matched_all_8000_streams":True,
        "L506_access":0,"external_memory_access":0,"neighbor_NDCT_access":0,
        "official_rddm_source_sha":source["official_rddm_commit"],"meddino_source_sha":source["meddino_commit"],
        "meddino_checkpoint_sha256":source["meddino_checkpoint_sha256"],
        "peak_vram_bytes":{a:final[a]["peak_vram_bytes"] for a in ARMS}}
    write_json(ROOT/"FINAL_SUMMARY.json",summary)
    lines=["# Official RDDM + 2.5D + MedDINOv3 Anatomy Condition V1","",
        "两臂从此前 Official RDDM 2D 8k EMA 初始化，再分别进行 8k 个 fresh-Adam continuation steps。Official RDDM ResBlock、8 个 LinearAttention、middle Full Attention 和双 skip topology 全部保留；diffusion、pred_res MSE、sampler 与评价函数原样复用。","",
        "## 起点与实现","",
        f"2D 8k 起点 EMA Fast64：PSNR {parent['psnr']:.6f} dB；SSIM {parent['ssim']:.6f}；RMSE {parent['rmse_hu']:.6f} HU；MAE {parent['mae_hu']:.6f} HU。",
        "2.5D stem 通道为 [x_t, prev LDCT, center LDCT, next LDCT]；两邻居权重初始为零，旧 x_t/center 权重和 bias 完整复制。逐患者相邻 slice、边界复制及固定 Fast64 manifest 来自已验证 Direct 实验，context_data.py 字节一致。",
        "MedDINO ViT-B/16 只接收 center LDCT（HU clip [-1000,1000]，mean 65.1084213256836，std 178.01663208007812，repeat 到三通道），官方 get_intermediate_layers(n=[2,5,8,11],norm=True,return_class_token=False,return_extra_tokens=False)。四个 768→64 独立 1×1 adapter 后 concat，256→128 1×1 + SiLU 得到共享 anatomy source。",
        "九个独立 residual affine modulation：E0/E1/D2/D3 global；E2/E3/MID/D0/D1 spatial。GN groups=8、affine=False，最后 gamma/beta projection zero-init；没有 Cross-Attention、额外 gate、sigmoid/tanh 或新 loss。每处均在原生 Attention 前，原生 Attention 参数和代码保持原样。",
        f"实际 batch={summary['batch_config']['batch']}、accum={summary['batch_config']['accum']}，effective batch=2。Adam lr 2e-4，betas=(0.9,0.99)，wd=1e-9；EMA=0.995；clip=1；AMP=False；seed=20260829。训练 L067/L143；评价 L333 Fast64，HU window [-1000,1000]，PSNR data_range=2000 HU。",
        "代码入口：model.py、med_loader.py、train.py、evaluate.py、readiness.py、postprocess.py；官方 diffusion 完整副本为 src/official_rddm_ct.py。源码、父 checkpoint 与 loader/配对实现身份见 SOURCE_PROVENANCE.json；运行环境和命令见 ENVIRONMENT.json、RUN_MANIFEST.json。","",
        "## Matched EMA Fast64","",
        "| Step | Control PSNR / SSIM / RMSE HU / MAE HU | Candidate PSNR / SSIM / RMSE HU / MAE HU | ΔPSNR dB |",
        "|---:|---|---|---:|"]
    for r in table:
        c=" / ".join(f"{r['control_'+m]:.6f}" for m in METRICS);n=" / ".join(f"{r['candidate_'+m]:.6f}" for m in METRICS)
        lines.append(f"| {r['step']} | {c} | {n} | {r['delta_psnr']:+.6f} |")
    lines+=["",f"8k ΔPSNR={last['delta_psnr']:+.6f} dB；PSNR 改善 {better}/64 slices（{better/64:.2%}）。完整四项 delta 和逐 slice 结果见 VALIDATION_HISTORY.csv、PER_SLICE_DELTA.csv。","",
        "## Condition 实际强度（8k EMA，两次 sampler forward ×64 slices 的均值）","",
        "| Stage | delta RMS | base RMS | mean delta/base | gamma RMS | beta RMS |","|---|---:|---:|---:|---:|---:|"]
    for r in activity:
        if r["step"]==8000:lines.append("| "+r["stage"]+" | "+" | ".join(f"{r[k]:.6g}" for k in ("delta_feature_rms","base_feature_rms","delta_base_ratio","gamma_rms","beta_rms"))+" |")
    ready=json_read(ROOT/"READINESS.json")
    lines+=["","全部节点见 CONDITION_ACTIVITY.csv。Native Attention 每个训练 step 的聚合梯度有限且非零，详见 TRAINING_DIAGNOSTICS.csv；readiness 对九处实际 forward 调用以及 Attention 输入为 condition 输出进行了 hook/指针验证。",
        f"readiness：2D→2.5D max_abs_diff={ready['two_dimensional_vs_two_point_five_max_abs_diff']:.9g}；Control→Candidate step0 max_abs_diff={ready['control_vs_candidate_step0_max_abs_diff']:.9g}。九处 step0 delta RMS=0；首 backward 九个 projection 梯度有限非零；optimizer step 后第二 backward 四个层 adapter、fusion 和九个 stage adapter 梯度有限非零。MedDINO grad=None、requires_grad=False、eval，训练后权重哈希不变。",
        f"两臂全部 8000 step 的样本、augmentation、timestep 和 noise stream hash 一致；first-batch 完整证据见 DATA_ACCESS_AUDIT.json。readiness GPU 峰值 {ready['peak_allocated_bytes']/1024**3:.3f} GiB，训练峰值 Control {final[ARMS[0]]['peak_vram_bytes']/1024**3:.3f} GiB、Candidate {final[ARMS[1]]['peak_vram_bytes']/1024**3:.3f} GiB。","",
        "## 数据边界与 promotion","",
        "L506 access=0；external memory access=0；neighbor NDCT access=0；blocked attempts=0。guard 按 patient/index/view 审核实际打开，监督仅 center NDCT。所有数据访问证据见 DATA_ACCESS_AUDIT.json。",
        f"Full-L333 gate（ΔPSNR≥0.03 dB 且改善≥60% slices）：{'PASS，已同时评估两臂 8k EMA 的 610 slices' if promotion else '未通过，未执行 Full-L333'}。"]
    if full:lines.append("Full-L333 指标及 delta："+json.dumps(full,ensure_ascii=False))
    if inference:
        lines+=["","## 免费推理诊断","", "| Mode | PSNR dB |","|---|---:|"]
        for mode,v in inference.items():lines.append(f"| {mode} | {v['psnr']:.6f} |")
        lines.append("CONDITION_OFF 将九处 delta 强制为零；SHUFFLED_MEDDINO 在固定 Fast64 的相邻二元 batch 中交换 frozen anatomy maps，RDDM 输入和逐 slice sampling seed 不变。诊断差值用于观察实际依赖程度，不改变 promotion 结果。")
    lines+=["","## 结果判断","",
        ("本次 matched Fast64 的 Candidate 获得正增益。" if last["delta_psnr"]>0 else "本次 matched Fast64 没有得到 Candidate PSNR 正增益。")+"结论仅针对单 seed、一个验证患者和这次 8k continuation。Condition 活动非零证明模块参与计算；是否改善重建由指标判断。若 NORMAL 与 OFF/SHUFFLED 差异很小，不能仅依据模块活动宣称 patient-specific anatomy 被有效利用。",
        "checkpoint、CT 图像、DICOM、NPY 医疗数据、MedDINO checkpoint 和训练缓存均不进入 Git。checkpoint 保存在忽略的 runs/<arm>/checkpoints；原始日志在 logs/ 和 runs/<arm>/train.jsonl。"]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print("FINAL_COMPLETE "+json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
