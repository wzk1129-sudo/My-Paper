from common import *
import csv
from mayo_data import FAST64, Mayo, VAL_PATIENTS
from metrics import window_metrics

@torch.no_grad()
def evaluate(model,step,out=OUT):
    data=Mayo(CFG["data_root"],VAL_PATIENTS)
    lookup={r[1]:i for i,r in enumerate(data.records)}
    assert all(i in lookup for i in FAST64)
    previous=model.training; model.eval(); rows=[]
    try:
        for index in FAST64:
            c,y=data.get(lookup[index])
            generator=torch.Generator(device="cuda").manual_seed(20260829+index)
            pred=model.sample(c[None].cuda(),generator)
            assert torch.isfinite(pred).all()
            pred_hu=((pred[0,0].cpu().numpy()+1)/2)*4096-1024
            target_hu=y[0].numpy()*4096-1024
            row={"patient_id":"L333","slice_index":index,"step":step}
            row.update(window_metrics(pred_hu,target_hu,-1000,1000))
            row.update({"soft_"+k:v for k,v in window_metrics(pred_hu,target_hu,-160,240).items()})
            assert all(np.isfinite(v) for k,v in row.items() if k!="patient_id")
            rows.append(row)
    finally: model.train(previous)
    with (out/f"validation_step_{step}.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    return {"step":step,**{k:float(np.mean([r[k] for r in rows])) for k in rows[0]
                         if k not in ("step","patient_id","slice_index")}}

if __name__=="__main__":
    import argparse
    from src.residual_diffusion import ResidualDiffusion
    ap=argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("--output",required=True)
    args=ap.parse_args(); setup()
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=ResidualDiffusion().cuda(); model.load_state_dict(ck["ema_state_dict"])
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    print(evaluate(model,ck["optimizer_step"],dest))
