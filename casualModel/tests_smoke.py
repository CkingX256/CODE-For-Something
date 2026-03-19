import numpy as np
import pandas as pd

from caxm import get_cam as get_caxm, cam_infer, cam_fit
from caxmpp import get_cam as get_caxmpp, cam_infer as campp_infer, campp_fit
from vcg import get_model as get_vcg, infer_y_from_features
from vcg_forest import get_model as get_vcgf, infer as infer_vcgf


def smoke_caxm():
    model = get_caxm(n=8, card=2, seed=0)
    rows = []
    for i in range(16):
        rows.append({'T': i % 2, 'y': np.array([0.8, 0.2]) if i % 2 == 0 else np.array([0.2, 0.8]), 'evidence': {'X0': i % 2}})
    cam_fit(model, rows, max_rows=16)
    p, _ = cam_infer(model, target='Y', do_dict={'T': 1}, evidence={'X0': 1})
    assert np.isfinite(np.asarray(p)).all()


def smoke_caxmpp():
    model = get_caxmpp(n=8, card=2, r=4, d=8, seed=0)
    rows = []
    for i in range(16):
        rows.append({'T': i % 2, 'y': np.array([0.75, 0.25]) if i % 2 == 0 else np.array([0.25, 0.75]), 'evidence': {'X0': i % 2}})
    campp_fit(model, rows, epochs=1, max_rows=16)
    p, _ = campp_infer(model, target='Y', do_dict={'T': 1}, evidence={'X0': 1})
    assert np.isfinite(np.asarray(p)).all()


def smoke_vcg():
    model = get_vcg(seed=0)
    p, _ = infer_y_from_features(model, {f'X{i}': 0 for i in range(25)}, do_t=1, eps_seed=0)
    assert np.isfinite(float(p))


def smoke_vcgf():
    model = get_vcgf(seed=0)
    p, _ = infer_vcgf(model, target='Y', do_dict={'T': 1}, evidence={f'X{i}': 0 for i in range(25)})
    assert np.isfinite(np.asarray(p)).all()


if __name__ == '__main__':
    smoke_caxm()
    smoke_caxmpp()
    smoke_vcg()
    smoke_vcgf()
    print('smoke tests passed')
