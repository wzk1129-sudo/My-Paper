"""EMA validation using the frozen L333 metric and fixed slice seeds."""
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from fast_dev.data import FastPairs, DataAccessGuard
from fast_dev.evaluate import validate as control_validate
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import METRICS, write_json
from .protocol import ARMS, experiment_config, make_model, HERE
from .runtime import direct_predict

@torch.no_grad()
def validate_arm(arm, model, dataset, cfg, device, rows_path=None):
    if arm == ARMS[0]:
        result = control_validate(model, dataset, cfg, device, rows_path)
        rows = json.loads(rows_path.read_text()) if rows_path else None
        return result, rows
    model.eval()
    rows = []
    started = time.perf_counter()
    for position in range(len(dataset)):
        ldct, target, meta = dataset.load(position)
        ldct = torch.from_numpy(ldct).unsqueeze(0).to(device)
        target = torch.from_numpy(target).unsqueeze(0).to(device)
        x_input = ldct * 2 - 1
        pred_res = direct_predict(model, ldct)
        prediction = ((x_input - pred_res).clamp(-1, 1) + 1) * 0.5
        assert prediction.shape == (1, 1, 512, 512) and torch.isfinite(prediction).all()
        values = slice_metrics(prediction, target, cfg)
        assert all(math.isfinite(values[k]) for k in METRICS)
        loss = float(F.mse_loss(pred_res, x_input - (target * 2 - 1)))
        assert math.isfinite(loss)
        rows.append({**meta, **{k: values[k] for k in METRICS}, "validation_loss": loss})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if rows_path:
        write_json(rows_path, rows)
    result = {k: float(np.mean([row[k] for row in rows])) for k in (*METRICS, "validation_loss")}
    return {**result, "slices": len(rows), "patient": "L333", "weights": "ema",
            "validation_mode": dataset.mode, "validation_seconds": time.perf_counter()-started}, rows

def load_ema(arm, checkpoint, device):
    cfg, _ = experiment_config()
    ck = torch.load(checkpoint, map_location="cpu")
    assert ck["arm"] == arm and ck["optimizer_step"] == 12000 and ck["config"] == cfg
    model = make_model(arm).to(device)
    model.load_state_dict(ck["ema_state_dict"], strict=True)
    model.eval()
    return model

def full_evaluate(device="cuda:0"):
    cfg, frozen = experiment_config()
    device = torch.device(device)
    by_arm = {}
    for arm in ARMS:
        folder = HERE / "runs" / arm
        ck = folder / "checkpoints" / "step_12000.pt"
        if not ck.is_file():
            raise FileNotFoundError(ck)
        model = load_ema(arm, ck, device)
        with DataAccessGuard(frozen, "FULL", allow_training=False, allow_memory=False) as guard:
            dataset = FastPairs(frozen, "validation", "FULL")
            summary, rows = validate_arm(arm, model, dataset, frozen, device,
                                         HERE / ("FULL_L333_" + arm + ".json"))
        access = guard.report()
        assert access["patients_opened"] == ["L333"]
        assert access["validation_indices_opened"] == list(range(610))
        assert access["test_opens"] == access["external_memory_opens"] == access["blocked_attempts"] == 0
        by_arm[arm] = {"summary": summary, "rows": rows, "access": access}
        write_json(HERE / ("FULL_L333_" + arm + "_ACCESS.json"), access)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return by_arm
