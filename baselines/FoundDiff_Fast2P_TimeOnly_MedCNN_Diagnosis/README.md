# FoundDiff Fast2P paired experiment and diagnosis

This archive contains actual executed source/configs and original reports for Fast2P_TimeOnly_Control_8k and Fast2P_TimeOnly_MedCNN_8k. DIAGNOSIS.md explains the 4k→8k regression, confirmed protocol inconsistencies and uncertainties.

- experiments/: exact executed source, training, data loader, frozen metrics, configs, aggregate validation summaries and original per-run reports.
- vendor/: DINOv3 runtime imports actually used, original notices and both relevant license files.
- training_curves/: 100-step aggregate loss and pre-clip global gradient statistics; LR reconstructed from source and optimizer checkpoint snapshots, not per-step LR logging.
- REPORT.md, paired_comparison.csv, FINAL_AUDIT.json: original paired report, aggregate comparison and audit.
- source_manifest.json: copied code/config hashes checked against final checkpoint source hashes.

No patient CT/DICOM/NPY, images, per-slice records, model checkpoints, pretrained weights, or large raw logs are included. Validation Fast64 indices are the already-published historical list at https://github.com/wzk1129-sudo/My-Paper/blob/adc8f138e1e02a1ac567177d6508436c2b5c6f36/baselines/FoundDiff_Lite_TimeDACB/experiments/FoundDiff_Lite_TimeDACB_8k/L333_fast64.txt ; place it in each experiment directory before reproduction.

MedDINO loader expects pretrained/MedDINOv3_ViTB16_CT3M_model.pth (not included). Official weight source https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M revision 20d233b0ce1b1f93c46550787cf4b409a6f4a95e SHA256 492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2. Vendor code derives from https://github.com/ricklisz/MedDINOv3 commit a2460c2547899283446e520b84523454f7e25608; upstream terms are preserved in vendor/.
