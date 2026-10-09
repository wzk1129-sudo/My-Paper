"""Finalize the saved-only result after the conditional Full-L333 gate."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    gate = json.loads((ROOT / "FULL_L333_GATE.json").read_text())
    report = ROOT / "REPORT.md"
    assert report.is_file()
    result = {"status": "COMPLETE", "full_l333_gate": gate}
    if gate["eligible"]:
        full = json.loads((ROOT / "FULL_L333_RESULT.json").read_text())
        metrics = full["metrics"]
        assert full["step"] == 8000 and full["mode"] == "FULL" and metrics["slices"] == 610
        access = full["access"]
        assert all(access[k] == 0 for k in ("L506_opens", "external_memory_opens", "neighbor_target_opens", "blocked_attempts"))
        content = report.read_text()
        assert "## Full-L333" not in content
        content += ("\n## Full-L333 gated validation\n\n"
                    f"A single authorized Full-L333 evaluation (610 slices) gave PSNR {metrics['psnr']:.6f} dB, "
                    f"SSIM {metrics['ssim']:.6f}, RMSE {metrics['rmse_hu']:.4f} HU, "
                    f"MAE {metrics['mae_hu']:.4f} HU. No L506 or external-memory data were opened.\n")
        report.write_text(content)
        result["full_l333_metrics"] = {k: metrics[k] for k in ("psnr", "ssim", "rmse_hu", "mae_hu", "slices")}
    else:
        result["full_l333_reason"] = "Fast64 threshold not met; Full-L333 was not run"
    assert not (ROOT / "PIPELINE_COMPLETE.json").exists()
    (ROOT / "PIPELINE_COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PIPELINE_COMPLETE", gate["eligible"])

if __name__ == "__main__":
    main()
