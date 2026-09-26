"""20-step isolated smoke and one permitted structure-loss scale calibration."""
import json
import math
import torch

from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, config
from .runtime import StructSupTrainState


def activity(block):
    return {k:float(v) for k,v in block.last_activity.items()}


def main():
    assert not (HERE/"SMOKE_RESULT_FINAL.json").exists()
    assert not (HERE/"LAMBDA_CALIBRATION_FINAL.json").exists()
    cfg=config()
    assert cfg["structure_supervision"]["lambda128"]==cfg["structure_supervision"]["lambda64"]==.05
    guard=ContextDataGuard(cfg,"FAST",allow_training=True)
    with guard:
        state=StructSupTrainState("cuda:0",guard)
        u=state.model.model.unet0
        rows=[]
        for _ in range(20):
            row=state.update()
            rows.append(row)
            if state.step%5==0:
                print("SMOKE",state.step,row["base_loss"],
                      row["structure_loss_128"],row["structure_loss_64"],
                      row["train_loss"],flush=True)
        access=guard.report()
        assert all(access[k]==0 for k in ("L506_opens","external_memory_opens",
                   "neighbor_target_opens","blocked_attempts"))
        assert state.nonfinite_count==0 and all(v>0 for v in state.grad_usage.values())
        assert all(math.isfinite(r[k]) for r in rows for k in (
                    "base_loss","structure_loss_128","structure_loss_64","image_aux_loss","train_loss"))
        assert all(not p.requires_grad and p.grad is None
                   for p in u.structure.anatomy.med.parameters())
        assert not u.structure.anatomy.med.training
        assert state.peak_vram < 20*2**30
        s=u.structure
        expected={"x1":[2,20,512,512],"x2":[2,40,256,256],
                  "x3":[2,80,128,128],"x4":[2,80,64,64],
                  "H128":[2,64,128,128],"H64":[2,128,64,64],
                  "S128":[2,64,128,128],"S64":[2,128,64,64]}
        assert s.last_shapes==expected
        assert s.last_predictions["128"].shape==(2,1,128,128)
        assert s.last_predictions["64"].shape==(2,1,64,64)
        assert state.last_aux["pred128_spatial_std"]>0
        assert state.last_aux["pred64_spatial_std"]>0
        assert all(float(b.last_activity["weighted_residual_rms"])>0
                   for b in (u.bca128,u.bca64))
        means={key:sum(r[key] for r in rows)/20 for key in (
            "base_loss","structure_loss_128","structure_loss_64",
            "weighted_structure_128","weighted_structure_64","image_aux_loss","weighted_image_aux","train_loss")}
        ratios=[means["weighted_structure_"+x]/means["base_loss"] for x in ("128","64")]
        if max(ratios)>.2:
            final=.02
            reason="at least one 0.05-weighted structure contribution exceeds 20% of base"
        elif max(ratios)<.01:
            final=.1
            reason="both 0.05-weighted structure contributions below 1% of base"
        else:
            final=.05
            reason="both structure contributions within calibration band"
        image_ratio=means["weighted_image_aux"]/means["base_loss"]
        if image_ratio>.2:
            image_final=.02
        elif image_ratio<.01:
            image_final=.1
        else:
            image_final=.05
        result={
            "status":"PASS","steps":20,"initialization":state.init_audit,
            "loss_means":means,"loss_first":rows[0],"loss_last":rows[-1],
            "gradient_usage":state.grad_usage,"shapes":s.last_shapes,
            "pred_structure_shapes":{"128":list(s.last_predictions["128"].shape),
                                     "64":list(s.last_predictions["64"].shape)},
            "structure_activity":{k:{a:float(v) for a,v in d.items()}
                                  for k,d in s.last_activity.items()},
            "target_stats":state.target_maker.last_stats,
            "beta128":float(s.beta128.detach()),"beta64":float(s.beta64.detach()),
            "alpha128":float(u.bca128.alpha.detach()),"alpha64":float(u.bca64.alpha.detach()),
            "bca128":activity(u.bca128),"bca64":activity(u.bca64),
            "train_stream_sha256":state.stream_digest.hex(),
            "seconds_per_step":state.train_seconds/20,
            "peak_vram_bytes":state.peak_vram,"nonfinite_count":state.nonfinite_count,
            "access":access,
        }
        write_json(HERE/"SMOKE_RESULT_FINAL.json",result)
        calibration={
            "status":"PASS","smoke_steps":20,
            "lambda128_initial":.05,"lambda64_initial":.05,
            "contribution_to_base_initial":{"128":ratios[0],"64":ratios[1]},
            "lambda128_final":final,"lambda64_final":final,
            "lambda_img_initial":.05,"lambda_img_final":image_final,
            "image_contribution_to_base_initial":image_ratio,
            "image_contribution_to_base_final_estimated":image_final*means["image_aux_loss"]/means["base_loss"],
            "estimated_contribution_to_base_final":{
                "128":final*means["structure_loss_128"]/means["base_loss"],
                "64":final*means["structure_loss_64"]/means["base_loss"]},
            "decision":reason,"image_auxiliary_loss":True,
            "formal_freeze":True,
        }
        write_json(HERE/"LAMBDA_CALIBRATION_FINAL.json",calibration)
        print(json.dumps({"status":"PASS","ratios":ratios,"final_lambda":final,"lambda_img":image_final,
                          "peak_vram_bytes":state.peak_vram}),flush=True)


if __name__=="__main__":
    main()
