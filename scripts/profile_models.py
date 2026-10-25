"""
Re-profile every model with ONE convention (Tables 4 and 7, Supplementary Table S1): parameters before and after
Conv-BN fusion and GFLOPs at 640x640 exactly as Ultralytics prints them, plus the module type at
each layer index so that the positions swapped in Tables 4 and 7 can be verified from the
checkpoints themselves.

    # YAMLs (architecture only)
    python profile_models.py --nc 6 --yaml <repo>/cfg/ablation/*.yaml <repo>/cfg/SCD-YOLO.yaml

    # trained checkpoints (what was actually evaluated) - recommended for Tables 3-9, 13, 14
    python profile_models.py --ckpt runs/*/weights/best.pt --layers 23 24 27 28

    # layer-by-layer table with output shapes (Supplementary Table S1)
    python profile_models.py --nc 6 --yaml <repo>/cfg/ablation/abl_S1_D1_A1.yaml --per-layer

Expected (nc=6, fused, YOLOv13 source tree of docs/INSTALL.md): YOLOv13n 2,467,497 | S0_D1 2,345,193 |
S1_D0 2,037,811 | SCD-YOLO 1,915,507 | ctrl_S1_ADown_A0 1,904,691 | ungated D1_A1 1,846,873 |
ungated D0_A0 1,969,177.
"""
import argparse
import csv
import glob
from copy import deepcopy


def describe(model, layers):
    out = {}
    for m in model.model:
        if getattr(m, "i", None) in layers:
            out[m.i] = type(m).__name__
    return out


def per_layer(model, imgsz=640):
    import torch
    rows, handles = [], []

    def hook(mod, inp, outp):
        shp = [tuple(o.shape[1:]) for o in outp if o is not None] if isinstance(outp, (list, tuple)) else [tuple(outp.shape[1:])]
        rows.append(dict(i=mod.i, f=mod.f, module=type(mod).__name__,
                         params=sum(p.numel() for p in mod.parameters()), out=shp))

    for m in model.model:
        handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    for h in handles:
        h.remove()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", nargs="*", default=[])
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--layers", type=int, nargs="*", default=[23, 24, 27, 28])
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--out", default="profile_models.csv")
    a = ap.parse_args()

    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    items = [("yaml", p) for g in a.yaml for p in sorted(glob.glob(g))] + \
            [("ckpt", p) for g in a.ckpt for p in sorted(glob.glob(g))]
    rows = []
    for kind, path in items:
        if kind == "yaml":
            m = DetectionModel(path, nc=a.nc, verbose=False)
        else:
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m = (ck.get("ema") or ck["model"]).float()
        m = deepcopy(m).cpu()
        if a.per_layer:
            print(f"\n=== {path}")
            for r in per_layer(deepcopy(m)):
                print(f"{r['i']:>3} {str(r['f']):>14} {r['module']:<22} {r['params']:>9,}  {r['out']}")
        p_unf = get_num_params(m)
        g_unf = get_flops(m, 640)
        mods = describe(m, set(a.layers))
        m.eval().fuse(verbose=False)
        p_f = get_num_params(m)
        g_f = get_flops(m, 640)
        rows.append(dict(source=path, nc=int(getattr(m, "nc", a.nc) or a.nc), params_unfused=p_unf, params_fused=p_f,
                         gflops_unfused=round(g_unf, 3), gflops_fused=round(g_f, 3),
                         layers=";".join(f"{k}:{v}" for k, v in sorted(mods.items()))))
        print(f"{path}\n   params unfused {p_unf:,} | fused {p_f:,} | GFLOPs {g_unf:.2f} (fused {g_f:.2f}) | {rows[-1]['layers']}")
    if rows:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
