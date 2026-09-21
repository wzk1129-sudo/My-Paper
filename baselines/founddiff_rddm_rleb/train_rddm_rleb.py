"""Minimal paired training entry; explicit steps and new output directory required.
Example: python train_rddm_rleb.py --ndct /path/ndct.npy --ldct /path/ldct.npy \
    --output /path/new_run --steps 1000 --device cuda:0
Arrays must be aligned float NCHW .npy files with C=1 and values in [0,1].
This entry does not reproduce the authors' dataset split, augmentation or EMA.
Full training requires separate user authorization in the remote project.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.rddm_rleb import build_baseline

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ndct', required=True)
    p.add_argument('--ldct', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--gradient-accumulation', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=10)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.gradient_accumulation) < 1:
        p.error('steps, batch size and accumulation must be positive')
    ndct, ldct = (np.load(f, mmap_mode='r', allow_pickle=False) for f in (args.ndct, args.ldct))
    if ndct.shape != ldct.shape or ndct.ndim != 4 or ndct.shape[1] != 1 or ndct.shape[0] == 0 or any(s % 8 for s in ndct.shape[-2:]):
        p.error('aligned nonempty N x 1 x H x W arrays, H/W divisible by 8, required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_baseline(image_size=ndct.shape[-1]).to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    with (out / 'train.jsonl').open('x') as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(args.gradient_accumulation):
                indices = rng.integers(0, len(ndct), size=args.batch_size)
                pair = [torch.as_tensor(np.array(a[indices], dtype=np.float32), device=args.device) for a in (ndct, ldct)]
                if any(not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1 for x in pair):
                    raise ValueError('Training input must be finite and normalized to [0,1]')
                loss = model(pair)[0] / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step()
            record = json.dumps({'step': step, 'loss': total, 'gradient_norm': float(norm)})
            log.write(record + '\n')
            log.flush()
            print(record, flush=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': args.steps, 'config': vars(args)}, out / 'final.pt')

if __name__ == '__main__':
    main()
