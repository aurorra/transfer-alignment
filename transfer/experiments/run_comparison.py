import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
import yaml
from codecarbon import OfflineEmissionsTracker

from transfer.src.utils.io_utils import append_jsonl
from transfer.src.alignment.hyperparameter_search import run_kpca_validation, run_ot_validation
from transfer.src.alignment.pca_adaptation import fit_pca, pca_align_and_backproject, kpca_align_and_preimage
from transfer.src.alignment.coral_adaptation import fit_coral, coral_transform
from transfer.src.alignment.ot_adaptation import ot_map_target_to_source
from transfer.src.classification.train_classifier import run_classification
from transfer.src.embedding.extractor import extract_cls_embeddings


# Set project root
project_root = Path(__file__).resolve().parents[2]
print(f'Root directory: {project_root}')


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    default_config_path = project_root / 'configs' / 'default.yaml'

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=default_config_path)
    config_args, _ = config_parser.parse_known_args()

    config = load_config(config_args.config)
    experiment_cfg = config['experiment']
    kpca_cfg = config['kpca']['selected']
    ot_cfg = config['ot']['selected']

    parser = argparse.ArgumentParser(
        description='Run representation-alignment transfer experiments.',
        parents=[config_parser],
    )
    parser.add_argument(
        '--aligners',
        nargs='+',
        choices=['pca', 'kpca', 'coral', 'ot'],
        default=experiment_cfg['aligners'],
        help='Alignment methods to run.',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=experiment_cfg['seeds'],
        help='Random seeds to use.',
    )
    parser.add_argument(
        '--search',
        nargs='+',
        choices=['kpca', 'ot'],
        default=experiment_cfg.get('search', []),
        help='Run the paper hyperparameter search for KPCA and/or OT on Case -> Verb only.',
    )
    parser.add_argument(
        '--extract-embeddings',
        action=argparse.BooleanOptionalAction,
        default=experiment_cfg.get('extract_embeddings', False),
        help='Force re-extraction of embeddings even if matching embeddings already exist.',
    )
    parser.add_argument(
        '--include-raw-baseline',
        action=argparse.BooleanOptionalAction,
        default=experiment_cfg.get('include_raw_baseline', True),
        help='Evaluate direct transfer without alignment.',
    )

    parser.add_argument(
        '--example',
        action='store_true',
        help='Run the lightweight Case -> Verb example using precomputed embeddings from the example folder.',
    )

    # KPCA selected settings, CLI values override the values from the config
    parser.add_argument('--kpca-gamma', type=float, default=kpca_cfg['gamma'])
    parser.add_argument('--kpca-alpha', type=float, default=kpca_cfg['alpha'])
    parser.add_argument('--kpca-n-fit', type=int, default=kpca_cfg['n_fit'])
    parser.add_argument('--kpca-n-components', type=int, default=kpca_cfg['n_components'])
    parser.add_argument('--kpca-k-align', type=int, default=kpca_cfg['k_align'])

    # OT selected settings, CLI values override the values from the config
    parser.add_argument('--ot-reg', type=float, default=ot_cfg['reg'])
    parser.add_argument(
        '--ot-standardize',
        action=argparse.BooleanOptionalAction,
        default=ot_cfg['standardize'],
    )

    return parser.parse_args(), config


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Enable deterministic algorithms when supported
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def start_tracker(output_dir: str, project_name: str, experiment_id: str, run_id: str):
    os.makedirs(output_dir, exist_ok=True)
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        experiment_id=experiment_id,
        output_dir=output_dir,
        output_file=f'{run_id}.csv',
        save_to_file=True,
        tracking_mode='machine',
        measure_power_secs=1,
        log_level='error',
    )
    tracker.start()
    return tracker


def stop_tracker(tracker: OfflineEmissionsTracker, csv_path: str, summary_path: str):
    emissions_kg = tracker.stop()

    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f'CodeCarbon did not create the expected CSV file: {csv_path}'
        )

    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError(f'CodeCarbon CSV is empty: {csv_path}')

    row = df.iloc[-1]

    duration_s = float(row.get('duration', np.nan))
    energy_kwh = float(row.get('energy_consumed', np.nan))
    gpu_energy_kwh = float(row.get('gpu_energy', np.nan))
    cpu_energy_kwh = float(row.get('cpu_energy', np.nan))
    ram_energy_kwh = float(row.get('ram_energy', np.nan))

    with open(summary_path, 'w') as f:
        f.write(f'emissions_kg={emissions_kg}\n')
        f.write(f'duration_s={duration_s}\n')
        f.write(f'energy_consumed_kwh={energy_kwh}\n')
        f.write(f'gpu_energy_kwh={gpu_energy_kwh}\n')
        f.write(f'cpu_energy_kwh={cpu_energy_kwh}\n')
        f.write(f'ram_energy_kwh={ram_energy_kwh}\n')

    return emissions_kg, duration_s, energy_kwh


def run_extract_embeddings(
    model_ckpt_path,
    tokenizer_name,
    data_path,
    output_path,
    extract,
    batch_size=256,
    device='cuda',
):
    output_path = Path(output_path)
    emb_path = output_path / 'embeddings.npy'
    labels_path = output_path / 'labels.npy'
    meta_path = output_path / 'embedding_meta.json'

    current_meta = {
        'model_ckpt_path': str(Path(model_ckpt_path).resolve()),
        'tokenizer_name': tokenizer_name,
        'data_path': str(Path(data_path).resolve()),
    }

    need_build = extract or (not emb_path.is_file()) or (not labels_path.is_file())

    if not need_build:
        if not meta_path.is_file():
            raise RuntimeError(
                f'Found existing embeddings in {output_path}, but no embedding_meta.json. '
                'Cannot verify whether they match the current run. '
                'Set --extract-embeddings once to rebuild them.'
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            old_meta = json.load(f)

        if old_meta != current_meta:
            raise RuntimeError(
                f'Existing embeddings in {output_path} do not match the current run.\n'
                f'Existing metadata: {old_meta}\n'
                f'Current metadata: {current_meta}\n'
                'Use --extract-embeddings or delete the old embedding folder.'
            )

    if need_build:
        print(f'Extracting embeddings from {data_path} on {device}')
        extract_cls_embeddings(
            model_ckpt_path=str(model_ckpt_path),
            tokenizer_name=tokenizer_name,
            data_path=str(data_path),
            output_path=str(output_path),
            batch_size=batch_size,
            device=device,
        )

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(current_meta, f, indent=2)

    if not emb_path.is_file():
        raise FileNotFoundError(f'Missing embeddings file: {emb_path}')
    if not labels_path.is_file():
        raise FileNotFoundError(f'Missing labels file: {labels_path}')

    X = np.load(emb_path)
    y = np.load(labels_path)

    if len(X) != len(y):
        raise ValueError(
            f'Embeddings/labels length mismatch in {output_path}: '
            f'{len(X)} embeddings vs {len(y)} labels.'
        )

    return X


def load_precomputed_embeddings(embedding_dir: Path):
    emb_path = embedding_dir / 'embeddings.npy'
    labels_path = embedding_dir / 'labels.npy'

    if not emb_path.is_file():
        raise FileNotFoundError(f'Missing embeddings file: {emb_path}')
    if not labels_path.is_file():
        raise FileNotFoundError(f'Missing labels file: {labels_path}')

    X = np.load(emb_path)
    y = np.load(labels_path)

    if len(X) != len(y):
        raise ValueError(
            f'Embeddings/labels length mismatch in {embedding_dir}: '
            f'{len(X)} embeddings vs {len(y)} labels.'
        )

    return X, emb_path, labels_path


def main():
    args, config = parse_args()

    if not args.example and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is not available. This experiment is expected to run on a GPU. '
            'Install a CUDA-enabled PyTorch build or use a compatible GPU environment.'
        )

    if args.example and args.search:
        raise ValueError('--example cannot be combined with --search.')

    # Main paths setup
    data_dir = project_root / 'data'
    results_dir = project_root / 'results'
    emb_dir = project_root / 'embeddings'

    experiment_cfg = config['experiment']
    kpca_config = config['kpca']
    ot_config = config['ot']
    example_config = config.get('example', {})

    extract_embeddings = args.extract_embeddings
    selection_metric = experiment_cfg['selection_metric']
    tokenizer = experiment_cfg['tokenizer']
    search_samples = int(experiment_cfg['search_samples'])

    # Fixed parameters for PCA and CORAL
    k = 'all'
    pca_projection = True
    coral_lam = 1e-3

    # Selected settings from the hyperparameter search, command-line arguments override the config
    kpca_params = {
        'gamma': args.kpca_gamma,
        'alpha': args.kpca_alpha,
        'n_fit': args.kpca_n_fit,
        'n_components': args.kpca_n_components,
        'k_align': args.kpca_k_align,
    }
    ot_params = {
        'metric': ot_config['selected']['metric'],
        'standardize': args.ot_standardize,
        'invert_standardization': args.ot_standardize,
        'reg': args.ot_reg,
        'numItermax': int(ot_config['selected']['num_iter_max']),
        'tol': float(ot_config['selected']['tol']),
    }

    search_methods = args.search
    include_raw_baseline = args.include_raw_baseline and not search_methods
    aligners = search_methods if search_methods else args.aligners

    if args.example:
        phase = 'example'
        results_dir = project_root / 'example' / 'results'
        random_seeds = [int(example_config.get('seed', 60))]
        transfer_pairs = [(
            example_config.get('source_type', 'case_error'),
            example_config.get('target_type', 'verb'),
        )]
        example_embedding_root = project_root / example_config.get(
            'embeddings_dir',
            'example/embeddings/finetuned_on_mbert_case_error',
        )
    else:
        phase = 'validation' if search_methods else 'test'
        random_seeds = args.seeds
        transfer_pairs = [
            (type_1, type_2)
            for type_1 in ['case_error', 'verb', 'cap']
            for type_2 in ['case_error', 'verb', 'cap']
            if type_1 != type_2
        ]

    # Go through each random seed and start a full run for all alignment methods and transfer directions
    for seed in random_seeds:
        print(f'\n===== Running seed {seed} =====')
        seed_tag = f'seed_{seed}'

        for aligner in aligners:
            for type_1, type_2 in transfer_pairs:

                # Hyperparameter search is performed only for Case -> Verb
                if search_methods and not (type_1 == 'case_error' and type_2 == 'verb'):
                    continue

                # Set seed for each run
                set_seed_all(seed)

                pair_tag = f'{type_1}_as_src_{type_2}_as_tgt'

                # Main output folder
                pair_results_dir = results_dir / aligner / pair_tag
                summary_dir = pair_results_dir / 'summary' / phase / seed_tag
                classifier_dir = pair_results_dir / 'classifier' / phase / seed_tag / 'MLP' / selection_metric
                cc_dir = pair_results_dir / 'codecarbon' / phase / seed_tag
                train_dir_type1 = classifier_dir / f'train_on_{type_1}'
                test_dir_type2 = classifier_dir / f'predict_{type_2}'

                if args.example:
                    emb_out_dir_type1_train = example_embedding_root / example_config.get('source_folder', 'case_error_cls_test')
                    emb_out_dir_type2_test = example_embedding_root / example_config.get('target_folder', 'verb_cls_test')
                    aligned_emb_dir = pair_results_dir / 'aligned_embeddings' / seed_tag
                else:
                    # Existing fine-tuned mBERT checkpoint
                    model_ckpt_path = (
                            project_root / 'results'
                            / f'finetuned_mbert_{type_1}'
                            / 'baseline'
                            / 'init_0'
                            / 'bert-base-multilingual-cased-final'
                    )

                    pair_emb_dir = emb_dir / aligner / pair_tag
                    before_alignment_dir = pair_emb_dir / 'before_alignment' / phase / seed_tag
                    after_alignment_dir = pair_emb_dir / 'after_alignment' / phase / seed_tag
                    emb_out_dir_type1_train = before_alignment_dir / f'{type_1}_src_train_val'
                    emb_out_dir_type2_test = before_alignment_dir / f'{type_2}_tgt_test'
                    aligned_emb_dir = after_alignment_dir

                    # During hyperparameter search, use the dedicated validation subset
                    if search_methods:
                        test_file_dir = str(data_dir / f'synthetic_data_{type_2}_V3_100000' / f'subset_n={search_samples}')
                    else:
                        test_file_dir = str(data_dir / f'synthetic_data_{type_2}_V3_100000' / 'test.csv')

                if search_methods:
                    validation_dir = pair_results_dir / 'validation_sweeps' / seed_tag
                    validation_dir.mkdir(parents=True, exist_ok=True)
                else:
                    aligned_emb_dir.mkdir(parents=True, exist_ok=True)
                    test_dir_type2.mkdir(parents=True, exist_ok=True)

                summary_dir.mkdir(parents=True, exist_ok=True)
                train_dir_type1.mkdir(parents=True, exist_ok=True)

                if not args.example:
                    emb_out_dir_type1_train.mkdir(parents=True, exist_ok=True)
                    emb_out_dir_type2_test.mkdir(parents=True, exist_ok=True)

                if aligner == 'pca' and not search_methods:
                    aligner_artifact_dir = pair_results_dir / 'alignment' / seed_tag / aligner
                    aligner_artifact_dir.mkdir(parents=True, exist_ok=True)

                results_summary_path = summary_dir / 'results.jsonl'
                train_ckpt_abs_type1 = train_dir_type1 / 'model.pt'

                run_id_base = aligner

                # Initialize variables
                best_ot, best_kpca = None, None
                baseline_metrics, kpca_metrics, ot_metrics, coral_metrics, pca_metrics = None, None, None, None, None
                extract_type_1_time, extract_type_2_time = None, None
                train_classifier_time, baseline_predict_time = None, None
                ot_align_time, ot_predict_time, ot_emb_path = None, None, None
                kpca_align_time, kpca_predict_time, kpca_emb_path = None, None, None
                pca_align_time, pca_predict_time = None, None
                coral_align_time, coral_predict_time = None, None
                tracker_full = None
                full_csv = None
                full_summary = None

                try:
                    ### ERROR TYPE-1 ###
                    tracker_full = start_tracker(
                        output_dir=str(cc_dir),
                        project_name='alignment_full_pipeline',
                        experiment_id=f'{type_1}_to_{type_2}',
                        run_id=f'{run_id_base}_FULL'
                    )
                    full_csv = f'{cc_dir}/{run_id_base}_FULL.csv'
                    full_summary = f'{cc_dir}/{run_id_base}_FULL_summary.txt'
                    t0 = time.time()

                    if args.example:
                        embeddings_type_1_train, source_embeddings_path, source_labels_path = load_precomputed_embeddings(
                            emb_out_dir_type1_train
                        )
                    else:
                        # Extract train/val embeddings of type-1
                        embeddings_type_1_train = run_extract_embeddings(
                            model_ckpt_path=str(model_ckpt_path),
                            tokenizer_name=tokenizer,
                            data_path=str(data_dir / f'synthetic_data_{type_1}_V3_100000'),
                            output_path=str(emb_out_dir_type1_train),
                            extract=extract_embeddings
                        )
                        source_embeddings_path = emb_out_dir_type1_train / 'embeddings.npy'
                        source_labels_path = emb_out_dir_type1_train / 'labels.npy'

                    extract_type_1_time = time.time() - t0
                    print(f'Load/extract type-1 embeddings: {extract_type_1_time:.3f} sec')

                    ### ERROR TYPE-2 ###
                    t0 = time.time()

                    if args.example:
                        embeddings_type_2_test, target_embeddings_path, target_labels_path = load_precomputed_embeddings(
                            emb_out_dir_type2_test
                        )
                    else:
                        # Extract test embeddings of type-2
                        embeddings_type_2_test = run_extract_embeddings(
                            model_ckpt_path=str(model_ckpt_path),
                            tokenizer_name=tokenizer,
                            data_path=test_file_dir,
                            output_path=str(emb_out_dir_type2_test),
                            extract=extract_embeddings
                        )
                        target_embeddings_path = emb_out_dir_type2_test / 'embeddings.npy'
                        target_labels_path = emb_out_dir_type2_test / 'labels.npy'

                    extract_type_2_time = time.time() - t0
                    print(f'Load/extract type-2 embeddings: {extract_type_2_time:.3f} sec')

                    # Train classifier on type-1 embeddings only if no reusable classifier exists
                    if train_ckpt_abs_type1.is_file():
                        print(f'Reusing existing source classifier: {train_ckpt_abs_type1}')
                        train_classifier_time = 0.0
                    else:
                        print(f'No existing source classifier found. Training new classifier: {train_ckpt_abs_type1}')
                        t0 = time.time()
                        run_classification(
                            embeddings_path=str(source_embeddings_path),
                            labels_path=str(source_labels_path),
                            output_path=str(train_dir_type1),
                            model_save_path=str(train_ckpt_abs_type1),
                            model_load_path='',
                            save_model=True,
                            mode='train',
                            selection_metric=selection_metric,
                            seed=seed,
                        )
                        train_classifier_time = time.time() - t0
                        print(f'Time for training classifier on {type_1}: {train_classifier_time:.3f} sec')

                    # Baseline: reuse type-1 classifier on raw type-2 test embeddings
                    if include_raw_baseline:
                        print(f'\nResults classifier 1 (trained on {type_1}) predictions on raw test set of {type_2}:')
                        t0 = time.time()
                        baseline_metrics = run_classification(
                            embeddings_path=str(target_embeddings_path),
                            labels_path=str(target_labels_path),
                            output_path=str(test_dir_type2 / 'raw'),
                            model_save_path='',
                            model_load_path=str(train_ckpt_abs_type1),
                            save_model=False,
                            mode='predict',
                            selection_metric=selection_metric,
                            seed=seed,
                        )
                        baseline_predict_time = time.time() - t0

                    if aligner == 'pca':
                        t0 = time.time()
                        # Fit PCA on type-1 train/val and type-2 test
                        pca1 = fit_pca(embeddings_type_1_train, n_components=k)
                        np.savez(
                            f'{aligner_artifact_dir}/pca_type1_k={k}.npz',
                            mean=pca1['mean'],
                            components=pca1['components'],
                            ev=pca1['explained_variance']
                        )

                        pca2 = fit_pca(embeddings_type_2_test, n_components=k)
                        np.savez(
                            f'{aligner_artifact_dir}/pca_type2_k={k}.npz',
                            mean=pca2['mean'],
                            components=pca2['components'],
                            ev=pca2['explained_variance']
                        )

                        # Align & back-project type-2 test into type-1 basis
                        X2_test_aligned = pca_align_and_backproject(
                            X_src=embeddings_type_2_test,  # type-2 test embeddings
                            pca_src=pca1,  # type-1 PCA (target basis)
                            pca_tgt=pca2,  # type-2 PCA (source basis)
                            k=k,  # rank
                            allow_scale=False,  # pure rotation
                            pca_space=pca_projection
                        )
                        pca_align_time = time.time() - t0
                        pca_emb_path = aligned_emb_dir / f'embeddings_pca_aligned_k={k}.npy'
                        np.save(pca_emb_path, X2_test_aligned)
                        print(f'PCA Alignment: {pca_align_time:.3f} sec')

                        # Predict with the type-1 classifier on aligned type-2 test
                        t0 = time.time()
                        print(f'\nResults classifier 1 (trained on {type_1}) predictions on test set of {type_2} aligned with PCA:')
                        pca_metrics = run_classification(
                            embeddings_path=str(pca_emb_path),
                            labels_path=str(target_labels_path),
                            output_path=str(test_dir_type2 / f'pca_aligned_k={k}'),
                            model_save_path='',
                            model_load_path=str(train_ckpt_abs_type1),
                            save_model=False,
                            mode='predict',
                            selection_metric=selection_metric,
                            seed=seed,
                        )
                        pca_predict_time = time.time() - t0
                        print(f'Predictions: {pca_predict_time:.3f} sec')

                    elif aligner == 'kpca':
                        y1_train = np.load(source_labels_path)

                        if search_methods:
                            kpca_val_dir = validation_dir / 'kpca' / f'val_n={search_samples}'
                            kpca_val_dir.mkdir(parents=True, exist_ok=True)

                            search_cfg = kpca_config['search']
                            best_kpca = run_kpca_validation(
                                embeddings_type_1_train=embeddings_type_1_train,
                                y1_train=y1_train,
                                embeddings_type_2_val=embeddings_type_2_test,
                                labels_path_val=str(target_labels_path),
                                train_ckpt_abs_type1=train_ckpt_abs_type1,
                                selection_metric=selection_metric,
                                out_dir=str(kpca_val_dir),
                                gamma_grid=search_cfg['gamma'],
                                alpha_grid=search_cfg['alpha'],
                                n_fit_grid=search_cfg['n_fit'],
                                n_components_grid=search_cfg['n_components'],
                                k_align_grid=search_cfg['k_align'],
                                seed=seed,
                            )

                            print('Best KPCA:', best_kpca)
                            if best_kpca.get('metrics') is not None:
                                print('Best KPCA metrics:', best_kpca['metrics'])

                        else:
                            best_kpca = kpca_params.copy()

                            t0 = time.time()
                            X2_test_aligned, meta = kpca_align_and_preimage(
                                embeddings_type_1_train,
                                y1_train,
                                embeddings_type_2_test,
                                n_fit=best_kpca['n_fit'],
                                n_components=best_kpca['n_components'],
                                k_align=best_kpca['k_align'],
                                alpha=best_kpca['alpha'],
                                gamma=best_kpca['gamma'],
                                seed=seed,
                            )
                            kpca_align_time = time.time() - t0
                            print('KPCA meta:', meta)

                            kpca_emb_path = (
                                    aligned_emb_dir
                                    / (
                                        f'embeddings_kpca_aligned'
                                        f'_d={best_kpca["n_components"]}'
                                        f'_kalign={best_kpca["k_align"]}'
                                        f'_gamma={best_kpca["gamma"]}'
                                        f'_alpha={best_kpca["alpha"]}'
                                        f'_nfit={best_kpca["n_fit"]}.npy'
                                    )
                            )
                            np.save(kpca_emb_path, X2_test_aligned)
                            print(f'KPCA Alignment: {kpca_align_time:.3f} sec')

                            print(
                                f'\nResults classifier 1 (trained on {type_1}) '
                                f'predictions on test set of {type_2} aligned with KPCA:'
                            )
                            t0 = time.time()
                            kpca_metrics = run_classification(
                                embeddings_path=str(kpca_emb_path),
                                labels_path=str(target_labels_path),
                                output_path=str(
                                    test_dir_type2
                                    / (
                                        f'kpca_aligned'
                                        f'_d={best_kpca["n_components"]}'
                                        f'_kalign={best_kpca["k_align"]}'
                                        f'_gamma={best_kpca["gamma"]}'
                                        f'_alpha={best_kpca["alpha"]}'
                                        f'_nfit={best_kpca["n_fit"]}'
                                    )
                                ),
                                model_save_path='',
                                model_load_path=str(train_ckpt_abs_type1),
                                save_model=False,
                                mode='predict',
                                selection_metric=selection_metric,
                                seed=seed,
                            )
                            kpca_predict_time = time.time() - t0
                            print(f'Predictions: {kpca_predict_time:.3f} sec')

                    elif aligner == 'coral':
                        # Fit CORAL transform
                        t0 = time.time()
                        coral_params = fit_coral(
                            X_src=embeddings_type_1_train,
                            X_tgt=embeddings_type_2_test,
                            lam=coral_lam
                        )

                        # Align type-2 test and evaluate with classifier 1 (trained on type-1)
                        X2_test_coral = coral_transform(embeddings_type_2_test, coral_params)
                        coral_align_time = time.time() - t0
                        coral_emb_path = aligned_emb_dir / 'embeddings_coral_aligned.npy'
                        np.save(coral_emb_path, X2_test_coral)
                        print(f'CORAL Alignment: {coral_align_time:.3f} sec')

                        print(
                            f'\nResults classifier 1 (trained on {type_1}) '
                            f'predictions on test set of {type_2} aligned with CORAL:'
                        )
                        t0 = time.time()
                        coral_metrics = run_classification(
                            embeddings_path=str(coral_emb_path),
                            labels_path=str(target_labels_path),
                            output_path=str(test_dir_type2 / 'coral_aligned'),
                            model_save_path='',
                            model_load_path=str(train_ckpt_abs_type1),
                            save_model=False,
                            mode='predict',
                            selection_metric=selection_metric,
                            seed=seed,
                        )
                        coral_predict_time = time.time() - t0

                    elif aligner == 'ot':
                        if search_methods:
                            ot_val_dir = validation_dir / 'ot'

                            search_cfg = ot_config['search']
                            best_ot = run_ot_validation(
                                embeddings_type_1_train=embeddings_type_1_train,
                                embeddings_type_2_val=embeddings_type_2_test,
                                labels_path_val=str(target_labels_path),
                                train_ckpt_abs_type1=train_ckpt_abs_type1,
                                selection_metric=selection_metric,
                                out_dir=str(ot_val_dir),
                                reg_grid=search_cfg['reg'],
                                standardize_grid=search_cfg['standardize'],
                                num_iter_max=int(ot_config['selected']['num_iter_max']),
                                metric=ot_config['selected']['metric'],
                                tol=float(ot_config['selected']['tol']),
                                seed=seed
                            )
                        else:
                            best_ot = ot_params.copy()

                            t0 = time.time()
                            X2_test_ot = ot_map_target_to_source(
                                Xs_train=embeddings_type_1_train,
                                Xt_test=embeddings_type_2_test,
                                reg=best_ot['reg'],
                                standardize=best_ot['standardize'],
                                invert_standardization=best_ot['invert_standardization'],
                                dtype=np.float32,
                                numItermax=best_ot['numItermax'],
                                tol=best_ot['tol'],
                                metric=best_ot['metric'],
                                seed=seed
                            )
                            ot_align_time = time.time() - t0

                            ot_emb_path = aligned_emb_dir / 'embeddings_ot_aligned.npy'
                            np.save(ot_emb_path, X2_test_ot)
                            print(f'Optimal Transport Alignment (Sinkhorn): {ot_align_time:.3f} sec')

                            print(
                                f'\nResults classifier 1 (trained on {type_1}) '
                                f'predictions on test set of {type_2} aligned with OT:'
                            )
                            t0 = time.time()
                            ot_metrics = run_classification(
                                embeddings_path=str(ot_emb_path),
                                labels_path=str(target_labels_path),
                                output_path=str(test_dir_type2 / 'ot_aligned'),
                                model_save_path='',
                                model_load_path=str(train_ckpt_abs_type1),
                                save_model=False,
                                mode='predict',
                                selection_metric=selection_metric,
                                seed=seed,
                            )
                            ot_predict_time = time.time() - t0
                    else:
                        raise ValueError("aligner must be 'pca', 'kpca', 'coral', or 'ot'")
                finally:
                    if tracker_full is not None and full_csv is not None and full_summary is not None:
                        cc_emissions_kg, cc_duration_s, cc_energy_kwh = stop_tracker(
                            tracker_full,
                            csv_path=full_csv,
                            summary_path=full_summary
                        )
                    else:
                        cc_emissions_kg, cc_duration_s, cc_energy_kwh = None, None, None

                # Persist one row per pair
                row = {
                    'type_1': type_1,
                    'type_2': type_2,
                    'aligner': aligner,
                    'classifier': 'MLP',
                    'selection_metric': selection_metric,
                    'baseline_raw': baseline_metrics,
                    'seed': seed,

                    'timing': {
                        'extract_type_1_time_sec': extract_type_1_time,
                        'extract_type_2_time_sec': extract_type_2_time,
                        'train_classifier_time_sec': train_classifier_time,
                        'baseline_predict_time_sec': baseline_predict_time,
                    },

                    'train_ckpt_abs_type1': train_ckpt_abs_type1,

                    'codecarbon': {
                        'emissions_kg': cc_emissions_kg,
                        'duration_s': cc_duration_s,
                        'energy_consumed_kwh': cc_energy_kwh,
                        'csv_path': full_csv,
                        'summary_path': full_summary
                    }
                }

                if aligner == 'ot':
                    row.update({
                        'ot_params': best_ot,
                        'ot_metrics': ot_metrics,
                        'ot_align_time_sec': ot_align_time,
                        'ot_predict_time_sec': ot_predict_time,
                        'ot_emb_path': ot_emb_path,
                    })
                    row['timing'].update({
                        'ot_align_time_sec': ot_align_time,
                        'ot_predict_time_sec': ot_predict_time,
                    })
                elif aligner == 'kpca':
                    row.update({
                        'kpca_params': best_kpca,
                        'kpca_metrics': kpca_metrics,
                        'kpca_align_time_sec': kpca_align_time,
                        'kpca_predict_time_sec': kpca_predict_time,
                        'kpca_emb_path': kpca_emb_path,
                    })
                    row['timing'].update({
                        'kpca_align_time_sec': kpca_align_time,
                        'kpca_predict_time_sec': kpca_predict_time,
                    })
                elif aligner == 'pca':
                    row.update({
                        'pca_metrics': pca_metrics,
                        'pca_align_time': pca_align_time,
                        'pca_predict_time': pca_predict_time,
                    })
                elif aligner == 'coral':
                    row.update({
                        'coral_metrics': coral_metrics,
                        'coral_align_time': coral_align_time,
                        'coral_predict_time': coral_predict_time,
                    })

                append_jsonl(str(results_summary_path), row)

                # Clean memory
                for name in [
                    'embeddings_type_1_train',
                    'embeddings_type_2_test',
                    'X2_test_aligned',
                    'X2_test_coral',
                    'X2_test_ot',
                    'pca1',
                    'pca2',
                    'coral_params',
                ]:
                    if name in locals():
                        del locals()[name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
