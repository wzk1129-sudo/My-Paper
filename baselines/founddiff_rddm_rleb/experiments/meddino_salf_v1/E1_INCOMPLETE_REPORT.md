# E1-SALF incomplete archival

The E1-SALF arm independently reached 6,000 optimizer steps and produced all planned Fast64 EMA metrics. Its 6k PSNR was 40.810963944 dB, versus 40.813155966 dB for matched D1 (delta -0.002192022 dB).

Checkpoint archival then failed because the new output directory lacked `checkpoints/`. No E1 6k checkpoint exists, and Full-L333 was not run. This is an incomplete experiment record, not a complete E1 model. Its existing source, metrics, queue status, and smoke audit are preserved. The smoke access audit reports zero L506, external-memory, neighbor-target, and blocked opens; a complete 6k access audit was not written before the checkpoint failure.

The later E12-HAF experiment must initialize independently from the canonical Direct checkpoint and must not load an E1 checkpoint.
