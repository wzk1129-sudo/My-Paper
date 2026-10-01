"""Synthetic acceptance only; never reads private images or model checkpoints."""
import argparse
import ast
import importlib.abc
import json
import sys
from pathlib import Path
import torch

FORBIDDEN = ('src.DADiff', 'src.DACLIP', 'src.emamba2', 'src.model_clipiqa',
             'open_clip', 'clip', 'lpips', 'timm', 'mamba_ssm', 'wandb', 'selective_scan')
class NoExternalModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in FORBIDDEN):
            raise AssertionError('Forbidden import: ' + fullname)
        return None

def no_load(*args, **kwargs):
    raise AssertionError('Checkpoint / pretrained weight loading forbidden in smoke test')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--full-resolution', action='store_true')
    parser.add_argument('--report', required=True)
    parser.add_argument('--upstream-source', help='Optional official DADiff.py for AST equivalence audit')
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(report_path)
    sys.meta_path.insert(0, NoExternalModels())
    torch.load = no_load
    torch.hub.load = no_load
    torch.hub.load_state_dict_from_url = no_load
    from src.rddm_rleb import build_baseline, ResnetBlock
    torch.set_num_threads(2)
    torch.manual_seed(10)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_baseline().to(device)
    net = model.model.unet0
    assert len(net.downs) == len(net.ups) == 4
    rlebs = [(name, module) for name, module in net.named_modules() if isinstance(module, ResnetBlock)]
    assert len(rlebs) == 10
    banned_names = ('mamba', 'ss2d', 'cssm', 'dacb', 'clip', 'dose', 'anatom', 'adaln', 'attention', 'prompt', 'text_mlp')
    for name, module in model.named_modules():
        assert not any(x in (name + type(module).__name__).lower() for x in banned_names), name
    traces = []
    hooks = []
    selected = {'init_conv', 'mid_block', 'final_res_block', 'final_conv'}
    for name, module in net.named_modules():
        if name in selected or name in [n for n, _ in rlebs] or (name.startswith(('downs.', 'ups.')) and len(name.split('.')) == 3):
            def record(mod, ins, out, name=name):
                traces.append({'name': name, 'in': list(ins[0].shape), 'out': list(out.shape)})
            hooks.append(module.register_forward_hook(record))
    ndct = torch.rand(2, 1, 64, 64, device=device)
    ldct = (ndct + 0.05 * torch.randn_like(ndct)).clamp(0, 1)
    loss = model([ndct, ldct])[0]
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert len([x for x in traces if x['name'] in [n for n, _ in rlebs]]) == 10
    batch_trace = list(traces)
    for hook in hooks:
        hook.remove()
    loss.backward()
    missing, nonfinite, zero = [], [], []
    for name, p in model.named_parameters():
        if p.grad is None: missing.append(name)
        elif not torch.isfinite(p.grad).all(): nonfinite.append(name)
        elif not torch.count_nonzero(p.grad): zero.append(name)
    assert not missing and not nonfinite and not zero, (missing, nonfinite, zero)
    grad_norm = float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters())))
    assert grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        inputs = torch.cat((ndct * 2 - 1, ldct * 2 - 1), dim=1)
        pred0 = net(inputs, torch.zeros(2, device=device))
        pred1 = net(inputs, torch.full((2,), 500., device=device))
        assert pred0.shape == ndct.shape and torch.isfinite(pred0).all()
        time_delta = float((pred1 - pred0).abs().max())
        changed = inputs.clone()
        changed[:, 1] *= 0.5
        condition_delta = float((net(changed, torch.zeros(2, device=device)) - pred0).abs().max())
        assert time_delta > 1e-6 and condition_delta > 1e-6
        t = torch.tensor([10, 900], device=device)
        clean, cond = ndct * 2 - 1, ldct * 2 - 1
        noise = torch.randn_like(clean)
        residual = cond - clean
        xt = model.q_sample(clean, residual, t, noise)
        expected_xt = clean + model.alphas_cumsum[t, None, None, None] * residual + model.betas_cumsum[t, None, None, None] * noise
        assert torch.equal(xt, expected_xt)
        prediction = model.model(torch.cat((xt, cond), dim=1), [model.alphas_cumsum[t] * 1000, model.betas_cumsum[t] * 1000])[0]
        expected_loss = (prediction - residual).square().flatten(1).mean(1).mean()
        actual_loss = model.p_losses([clean, cond], t, noise)[0]
        assert torch.allclose(expected_loss, actual_loss, atol=1e-7)
        sampled = model.sample([ldct], last=True)
        assert len(sampled) == 2 and sampled[-1].shape == ldct.shape
        assert all(torch.isfinite(x).all() for x in sampled)
        assert sampled[-1].min() >= 0 and sampled[-1].max() <= 1
        full_shape = None
        if args.full_resolution:
            full = net(torch.rand(1, 2, 512, 512, device=device) * 2 - 1, torch.tensor([500.], device=device))
            assert full.shape == (1, 1, 512, 512) and torch.isfinite(full).all()
            full_shape = list(full.shape)
    assert not any(any(n == p or n.startswith(p + '.') for p in FORBIDDEN) for n in sys.modules)
    # Verify retained diffusion is AST-identical to upstream except unused LPIPS construction.
    equivalence = 'NOT CHECKED: no upstream source provided'
    if args.upstream_source:
        upstream = ast.parse(Path(args.upstream_source).read_text())
        new = ast.parse((Path(__file__).resolve().parent / 'src/rddm_rleb.py').read_text())
        original_class = next(n for n in upstream.body if isinstance(n, ast.ClassDef) and n.name == 'ResidualDiffusion')
        new_class = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == 'ResidualDiffusion')
        init = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        init.body = [n for n in init.body if not (isinstance(n, ast.Assign) and any(isinstance(a, ast.Attribute) and a.attr == 'perceploss' for a in n.targets)) and not (isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute) and isinstance(n.iter.func.value, ast.Attribute) and n.iter.func.value.attr == 'perceploss')]
        assert ast.dump(original_class) == ast.dump(new_class), 'Unintended diffusion changes'
        equivalence = 'PASS except removal of unused LPIPS allocation'
    result = {'status': 'PASS', 'device': str(device), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'parameters': sum(p.numel() for p in model.parameters()),
              'parameter_tensors': len(list(model.parameters())), 'rleb_count': len(rlebs),
              'rleb_positions': [name for name, _ in rlebs], 'encoder_skip_count': 4, 'stem_skip_count': 1,
              'batch_shape': list(ndct.shape), 'loss': float(loss.detach()), 'gradient_norm': grad_norm,
              'missing_gradients': missing, 'nonfinite_gradients': nonfinite, 'all_zero_gradient_tensors': zero,
              'timestep_max_delta': time_delta, 'ldct_max_delta': condition_delta,
              'sample_shape': list(sampled[-1].shape), 'sample_steps': 2, 'full_forward_shape': full_shape,
              'forbidden_imports_and_weight_loads': 'blocked; none attempted',
              'diffusion_ast_equivalence': equivalence,
              'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'module_types': sorted({type(m).__name__ for m in model.modules()}), 'forward_trace': batch_trace}
    report_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('forward_trace', 'module_types')}, indent=2))

if __name__ == '__main__':
    main()
