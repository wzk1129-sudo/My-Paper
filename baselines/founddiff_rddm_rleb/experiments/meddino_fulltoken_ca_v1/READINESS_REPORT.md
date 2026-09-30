# Full-token CA readiness

Status: PASS
Direct 12k EMA checkpoint SHA256: 151b7e58d2ca0d936be0842434e3d17809b325f144d0d48e47c8565287bcdcdf
MedDINO checkpoint SHA256: 492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2; official source commit: a2460c2547899283446e520b84523454f7e25608
Direct shared weights bitwise identical; step-0 max abs difference: 0
MedDINO normalized final-layer patch tokens: [2,1024,768], CLS/register excluded.
CA128: 16384 queries x 1024 keys; CA64: 4096 queries x 1024 keys; 4 heads x 64.
Gate gradients: [[-1.247212111366025e-07, 4.3832123708398285e-08], [-2.094087108162057e-07, -1.0535782024589935e-07]]; projection gradient norms: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.9091218189659642e-11, 1.3891447887826036e-11, 5.316212400852116e-11, 2.5520766055997512e-11, 4.108720030870927e-09, 1.3399329423791073e-09, 2.12887529826844e-09, 7.927505873972507e-10]]
Q chunk: 0; small unchunked/chunked max diff: 2.38418579e-07
Peak allocated/reserved: 4981305344/6647971840 bytes
Two real optimizer-step times: [3.1496984199620783, 0.373354515992105]
Data access: {'mode': 'FAST', 'allowed_train_patients': ['L067', 'L143'], 'training_patients_opened': ['L067', 'L143'], 'validation_center_indices_opened': [], 'validation_ldct_indices_opened': [], 'center_target_opens': 0, 'center_ldct_opens': 0, 'neighbor_ldct_opens': 0, 'neighbor_target_opens': 0, 'training_target_opens': 4, 'total_opens': 16, 'L506_opens': 0, 'external_memory_opens': 0, 'blocked_attempts': 0, 'data_directory_enumeration': False}
