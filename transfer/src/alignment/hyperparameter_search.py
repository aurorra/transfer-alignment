import json
import os
import time

import numpy as np
from tqdm import tqdm

from transfer.src.utils.io_utils import to_jsonable
from transfer.src.alignment.ot_adaptation import ot_map_target_to_source
from transfer.src.alignment.pca_adaptation import kpca_align_and_preimage
from transfer.src.classification.train_classifier import run_classification


def run_ot_validation(
    embeddings_type_1_train,
    embeddings_type_2_val,
    labels_path_val,
    train_ckpt_abs_type1,
    selection_metric,
    out_dir,
    reg_grid,
    standardize_grid,
    num_iter_max=150,
    metric='sqeuclidean',
    tol=1e-4,
    seed: int = 64,
):
    """
    Grid-search the OT settings used in the paper on a target validation set.

    The searched parameters are:
      - regularization lambda
      - standardization

    All other OT settings are fixed.
    """
    try:
        os.makedirs(out_dir, exist_ok=False)
    except FileExistsError:
        print(f'Output dir already exists: {out_dir}')
        exit(-1)

    results_path = os.path.join(out_dir, 'ot_val_results.jsonl')
    best_path = os.path.join(out_dir, 'ot_val_best.json')

    best = {
        'score': -1.0,
        'reg': None,
        'standardize': None,
        'invert_standardization': None,
        'numItermax': int(num_iter_max),
        'metric': metric,
        'tol': float(tol),
        'metrics': None,
        'emb_path': None,
    }

    total = len(standardize_grid) * len(reg_grid)
    pbar = tqdm(total=total, desc='OT validation', unit='run')

    for std in standardize_grid:
        for reg in reg_grid:
            t0 = time.time()

            X_val_ot = ot_map_target_to_source(
                Xs_train=embeddings_type_1_train,
                Xt_test=embeddings_type_2_val,
                reg=float(reg),
                standardize=bool(std),
                invert_standardization=bool(std),
                dtype=np.float32,
                numItermax=int(num_iter_max),
                tol=float(tol),
                metric=metric,
                seed=seed,
            )

            emb_path = os.path.join(
                out_dir,
                f'emb_val_ot_{metric}_std{int(std)}_reg{reg:g}.npy'
            )
            np.save(emb_path, X_val_ot)

            clf_metrics = run_classification(
                embeddings_path=emb_path,
                labels_path=labels_path_val,
                output_path=os.path.join(
                    out_dir,
                    f'val_pred_{metric}_std{int(std)}_reg{reg:g}'
                ),
                model_save_path='',
                model_load_path=str(train_ckpt_abs_type1),
                save_model=False,
                mode='predict',
                selection_metric=selection_metric,
                seed=seed,
            )

            dt = time.time() - t0
            score = float(clf_metrics[selection_metric])

            row = {
                'reg': float(reg),
                'standardize': bool(std),
                'invert_standardization': bool(std),
                'numItermax': int(num_iter_max),
                'metric': metric,
                'tol': float(tol),
                'score': score,
                'metrics': clf_metrics,
                'time_sec': dt,
                'emb_path': emb_path,
            }

            if score > best['score']:
                best.update({
                    'score': score,
                    'reg': float(reg),
                    'standardize': bool(std),
                    'invert_standardization': bool(std),
                    'metrics': clf_metrics,
                    'emb_path': emb_path,
                })

            tqdm.write(
                f'[OT-VAL] metric={metric} std={int(std)} reg={reg:g} '
                f'{selection_metric}={score:.4f} time={dt:.1f}s'
            )

            pbar.set_postfix({
                'std': int(std),
                'reg': f'{reg:g}',
                selection_metric: f'{score:.4f}',
                'best': f'{float(best["score"]):.4f}',
            })
            pbar.update(1)

            with open(results_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(to_jsonable(row)) + '\n')

            with open(best_path, 'w', encoding='utf-8') as f:
                json.dump(to_jsonable(best), f, indent=2)

    pbar.close()
    tqdm.write(f'\nBest OT params: {best}')
    return best


def run_kpca_validation(
    embeddings_type_1_train,
    y1_train,
    embeddings_type_2_val,
    labels_path_val,
    train_ckpt_abs_type1,
    selection_metric,
    out_dir,
    gamma_grid,
    alpha_grid,
    n_fit_grid,
    n_components_grid,
    k_align_grid,
    seed,
):
    """
    Grid-search KPCA alignment hyperparams on a validation target set.

    Resume behavior:
      - If out_dir/kpca_val_results.jsonl exists, already-attempted configs are skipped.
      - Results are appended to JSONL.
      - Best-so-far is stored in out_dir/kpca_val_best.json (overwritten when improved).
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f'Using output dir: {out_dir}')

    results_path = os.path.join(out_dir, 'kpca_val_results.jsonl')
    best_path = os.path.join(out_dir, 'kpca_val_best.json')

    # ---- load done configs from existing results ----
    done = set()
    if os.path.isfile(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if all(k in r for k in ['gamma', 'alpha', 'n_fit', 'n_components', 'k_align']):
                    done.add((
                        float(r['gamma']),
                        float(r['alpha']),
                        int(r['n_fit']),
                        int(r['n_components']),
                        int(r['k_align']),
                    ))

    print(f'Found {len(done)} completed configs in {results_path}' if done else 'No existing results found.')

    # ---- load best (if present) ----
    best = {
        'score': -1.0,
        'gamma': None,
        'alpha': None,
        'n_fit': None,
        'n_components': None,
        'k_align': None,
        'metrics': None,
        'time_sec': None,
        'emb_path': None,
    }

    if os.path.isfile(best_path):
        try:
            with open(best_path, 'r', encoding='utf-8') as f:
                best = json.load(f)
            print(f"Loaded best so far: {best.get('score', -1.0)}")
        except Exception:
            print('Best file exists but could not be read; continuing with score=-1.0.')

    # ---- compute remaining for progress bar ----
    remaining = 0
    for g in gamma_grid:
        for a in alpha_grid:
            for n_fit in n_fit_grid:
                for n_components in n_components_grid:
                    for k_align in k_align_grid:
                        if k_align > n_components:
                            continue
                        key = (float(g), float(a), int(n_fit), int(n_components), int(k_align))
                        if key not in done:
                            remaining += 1

    pbar = tqdm(total=remaining, desc='KPCA validation (resume)', unit='run')

    for g in gamma_grid:
        for a in alpha_grid:
            for n_fit in n_fit_grid:
                for n_components in n_components_grid:
                    for k_align in k_align_grid:
                        if k_align > n_components:
                            continue

                        key = (float(g), float(a), int(n_fit), int(n_components), int(k_align))
                        if key in done:
                            continue

                        t0 = time.time()

                        # ---- KPCA align + preimage ----
                        try:
                            X_val_kpca, meta = kpca_align_and_preimage(
                                embeddings_type_1_train,
                                y1_train,
                                embeddings_type_2_val,
                                n_fit=int(n_fit),
                                n_components=int(n_components),
                                k_align=int(k_align),
                                alpha=float(a),
                                gamma=float(g),
                                seed=seed,
                            )
                        except Exception as e:
                            dt = time.time() - t0
                            tqdm.write(
                                f'[KPCA-VAL] SKIP ({type(e).__name__}): gamma={g:g} alpha={a:g} '
                                f'n_fit={n_fit} n_comp={n_components} k_align={k_align} msg={e}'
                            )

                            row = {
                                'status': 'skip',
                                'gamma': float(g),
                                'alpha': float(a),
                                'n_fit': int(n_fit),
                                'n_components': int(n_components),
                                'k_align': int(k_align),
                                'error': repr(e),
                                'time_sec': float(dt),
                            }
                            with open(results_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(to_jsonable(row)) + '\n')

                            done.add(key)
                            pbar.update(1)
                            continue

                        # ---- save embeddings ----
                        emb_path = os.path.join(
                            out_dir,
                            f'emb_val_kpca_gamma={g:g}_alpha={a:g}_n_fit={n_fit}_'
                            f'n_components={n_components}_k_align={k_align}.npy'
                        )
                        np.save(emb_path, X_val_kpca)

                        # ---- evaluate ----
                        clf_metrics = run_classification(
                            embeddings_path=emb_path,
                            labels_path=labels_path_val,
                            output_path=os.path.join(
                                out_dir,
                                f'val_pred_gamma={g:g}_alpha={a:g}_n_fit={n_fit}_'
                                f'n_components={n_components}_k_align={k_align}'
                            ),
                            model_save_path='',
                            model_load_path=str(train_ckpt_abs_type1),
                            save_model=False,
                            mode='predict',
                            selection_metric=selection_metric,
                            seed=seed,
                        )

                        dt = time.time() - t0
                        score = float(clf_metrics[selection_metric])

                        # ---- append result ----
                        row = {
                            'status': 'ok',
                            'gamma': float(g),
                            'alpha': float(a),
                            'n_fit': int(n_fit),
                            'n_components': int(n_components),
                            'k_align': int(k_align),
                            'score': float(score),
                            'metrics': clf_metrics,
                            'time_sec': float(dt),
                            'emb_path': emb_path,
                            'meta': meta,
                        }
                        with open(results_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(to_jsonable(row)) + '\n')

                        # ---- update best ----
                        if score > float(best.get('score', -1.0)):
                            best = {
                                'score': float(score),
                                'gamma': float(g),
                                'alpha': float(a),
                                'n_fit': int(n_fit),
                                'n_components': int(n_components),
                                'k_align': int(k_align),
                                'metrics': clf_metrics,
                                'time_sec': float(dt),
                                'emb_path': emb_path,
                            }
                            with open(best_path, 'w', encoding='utf-8') as f:
                                json.dump(to_jsonable(best), f, indent=2)

                        done.add(key)

                        tqdm.write(
                            f'[KPCA-VAL] gamma={g:g} alpha={a:g} n_fit={n_fit} '
                            f'n_components={n_components} k_align={k_align} '
                            f'{selection_metric}={score:.4f} time={dt:.1f}s'
                        )

                        pbar.set_postfix({
                            'gamma': f'{g:g}',
                            'alpha': f'{a:g}',
                            'n_fit': n_fit,
                            'n_comp': n_components,
                            'k_align': k_align,
                            selection_metric: f'{score:.4f}',
                            'best': f"{float(best.get('score', -1.0)):.4f}",
                        })
                        pbar.update(1)

    pbar.close()
    tqdm.write(f'\nBest KPCA params: {best}')
    return best
