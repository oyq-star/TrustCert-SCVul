"""Round 3 fixes: grouped CV, interpretability case study, EBM vs baselines explanation comparison."""
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trustcert_scvul.data.ingest import (
    load_smartbugs_curated, deduplicate, expand_with_negatives, TARGET_VULNS
)
from trustcert_scvul.features.structural import extract_features_batch
from trustcert_scvul.features.analyzer import (
    real_slither_to_analyzer_df, compute_consensus_features
)
from trustcert_scvul.models.train import (
    get_feature_columns, get_structural_only_cols
)
from trustcert_scvul.calibration.conformal import calibrate_conformal, evaluate_selective
from trustcert_scvul.analyzers.slither_runner import (
    run_slither_batch, findings_to_features, _load_version_map
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except ImportError:
    HAS_EBM = False


def grouped_kfold_splits(df, n_splits=5, seed=42):
    """Create grouped K-fold splits by source_hash."""
    rng = np.random.RandomState(seed)
    unique_hashes = list(df['source_hash'].unique())
    rng.shuffle(unique_hashes)
    fold_size = len(unique_hashes) // n_splits
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else len(unique_hashes)
        test_hashes = set(unique_hashes[start:end])
        remaining = [h for h in unique_hashes if h not in test_hashes]
        n_rem = len(remaining)
        cal_end = int(n_rem * 0.15)
        cal_hashes = set(remaining[:cal_end])
        train_hashes = set(remaining[cal_end:])
        folds.append((train_hashes, cal_hashes, test_hashes))
    return folds


def run_round3_fixes(base_dir=None, seed=42):
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent.parent.parent)

    raw_dir = os.path.join(base_dir, 'data', 'raw')
    curated_dir = os.path.join(raw_dir, 'smartbugs_curated', 'dataset')

    print("=" * 60)
    print("Round 3 Fixes")
    print("=" * 60)

    df = load_smartbugs_curated(os.path.join(raw_dir, 'smartbugs_curated'))
    df = expand_with_negatives(df)
    df = deduplicate(df)

    unique_contracts = df.drop_duplicates(subset=['contract_id', 'source_hash'])
    feat_df = extract_features_batch(unique_contracts)

    version_map = _load_version_map(curated_dir)
    print("\nRunning Slither...")
    slither_findings = run_slither_batch(unique_contracts, data_dir=curated_dir,
                                         version_map=version_map)
    slither_feat_df = findings_to_features(slither_findings, unique_contracts)

    # ============================================
    # FIX 1: Grouped 5-fold CV
    # ============================================
    print("\n" + "=" * 60)
    print("FIX 1: Grouped 5-Fold Cross-Validation")
    print("=" * 60)

    folds = grouped_kfold_splits(df, n_splits=5, seed=seed)
    cv_results = []

    for vuln_type in ['reentrancy', 'dos']:
        print(f"\n--- {vuln_type.upper()} ---")
        fold_metrics = defaultdict(list)

        for fold_idx, (train_h, cal_h, test_h) in enumerate(folds):
            df_fold = df.copy()
            df_fold['split'] = df_fold['source_hash'].apply(
                lambda h: 'train' if h in train_h else ('cal' if h in cal_h else 'test')
            )

            analyzer_df = real_slither_to_analyzer_df(slither_feat_df, df_fold, seed=seed+fold_idx)
            analyzer_df = compute_consensus_features(analyzer_df)

            meta_cols = ['contract_id', 'source_hash', 'vulnerability_type', 'label', 'split', 'dataset']
            full_df = df_fold[meta_cols].merge(
                feat_df, on=['contract_id', 'source_hash'], how='left'
            ).merge(
                analyzer_df, on=['contract_id', 'source_hash'], how='left'
            ).fillna(0)

            vt_df = full_df[full_df['vulnerability_type'] == vuln_type]
            train = vt_df[vt_df['split'] == 'train']
            cal = vt_df[vt_df['split'] == 'cal']
            test = vt_df[vt_df['split'] == 'test']

            if len(test) < 5 or (test['label'] == 1).sum() < 1:
                continue

            feature_cols = get_feature_columns(vt_df, vuln_type)
            X_train = np.nan_to_num(train[feature_cols].values.astype(float), nan=0.0)
            y_train = train['label'].values
            X_cal = np.nan_to_num(cal[feature_cols].values.astype(float), nan=0.0)
            y_cal = cal['label'].values
            X_test = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)
            y_test = test['label'].values

            models = {}
            if HAS_EBM:
                ebm = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                                    outer_bags=8, inner_bags=4,
                                                    random_state=seed)
                ebm.fit(X_train, y_train)
                models['TrustCert-EBM'] = ebm

            rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                        random_state=seed, class_weight='balanced')
            rf.fit(X_train, y_train)
            models['RandomForest'] = rf

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_cal_s = scaler.transform(X_cal)
            X_test_s = scaler.transform(X_test)

            lr = LogisticRegression(penalty='l1', solver='saga', max_iter=1000,
                                    random_state=seed, class_weight='balanced')
            lr.fit(X_train_s, y_train)
            models['L1-Logistic'] = lr

            for model_name, model in models.items():
                if model_name == 'L1-Logistic':
                    Xc, Xt = X_cal_s, X_test_s
                else:
                    Xc, Xt = X_cal, X_test

                y_prob = model.predict_proba(Xt)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                cal_probs = model.predict_proba(Xc)[:, 1]

                f1 = f1_score(y_test, y_pred, zero_division=0)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)

                q_hat = calibrate_conformal(y_cal, cal_probs, alpha=0.10)
                sel = evaluate_selective(y_test, y_prob, q_hat)

                fold_metrics[f'{model_name}_f1'].append(f1)
                fold_metrics[f'{model_name}_prec'].append(prec)
                fold_metrics[f'{model_name}_acc_prec'].append(sel['accepted_precision'])
                fold_metrics[f'{model_name}_accept_rate'].append(sel['acceptance_rate'])
                fold_metrics[f'{model_name}_n_accepted'].append(int(round(sel['acceptance_rate'] * len(y_test))))
                fold_metrics[f'{model_name}_n_test'].append(len(y_test))
                fold_metrics[f'{model_name}_coverage'].append(sel.get('empirical_coverage', 0.0))

        print(f"  {'Model':<15} {'F1 (mean±std)':>18} {'Prec (mean±std)':>18} {'AccPrec (mean±std)':>20}")
        for model_name in ['TrustCert-EBM', 'RandomForest', 'L1-Logistic']:
            f1_vals = fold_metrics.get(f'{model_name}_f1', [])
            prec_vals = fold_metrics.get(f'{model_name}_prec', [])
            acc_prec_vals = fold_metrics.get(f'{model_name}_acc_prec', [])
            if f1_vals:
                accept_vals = fold_metrics.get(f'{model_name}_accept_rate', [])
                n_accepted_vals = fold_metrics.get(f'{model_name}_n_accepted', [])
                n_test_vals = fold_metrics.get(f'{model_name}_n_test', [])
                coverage_vals = fold_metrics.get(f'{model_name}_coverage', [])
                pooled_n_test = sum(n_test_vals)
                pooled_n_accepted = sum(n_accepted_vals)
                pooled_acceptance = pooled_n_accepted / pooled_n_test if pooled_n_test else 0.0
                print(f"  {model_name:<15} {np.mean(f1_vals):.3f}±{np.std(f1_vals):.3f}"
                      f"       {np.mean(prec_vals):.3f}±{np.std(prec_vals):.3f}"
                      f"         {np.mean(acc_prec_vals):.3f}±{np.std(acc_prec_vals):.3f}"
                      f"   accept={np.mean(accept_vals):.1%}±{np.std(accept_vals):.1%}"
                      f"   n_acc={pooled_n_accepted}/{pooled_n_test}")
                cv_results.append({
                    'vulnerability': vuln_type,
                    'model': model_name,
                    'f1_mean': round(np.mean(f1_vals), 3),
                    'f1_std': round(np.std(f1_vals), 3),
                    'prec_mean': round(np.mean(prec_vals), 3),
                    'prec_std': round(np.std(prec_vals), 3),
                    'acc_prec_mean': round(np.mean(acc_prec_vals), 3),
                    'acc_prec_std': round(np.std(acc_prec_vals), 3),
                    'accept_rate_mean': round(np.mean(accept_vals), 3),
                    'accept_rate_std': round(np.std(accept_vals), 3),
                    'coverage_mean': round(np.mean(coverage_vals), 3),
                    'pooled_n_accepted': int(pooled_n_accepted),
                    'pooled_n_test': int(pooled_n_test),
                    'pooled_acceptance': round(pooled_acceptance, 3),
                    'per_fold_accept_rate': [round(x, 3) for x in accept_vals],
                    'per_fold_acc_prec': [round(x, 3) for x in acc_prec_vals],
                    'per_fold_n_accepted': [int(x) for x in n_accepted_vals],
                    'n_folds': len(f1_vals),
                })

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(base_dir, 'artifacts', 'grouped_cv_results.csv'), index=False)

    # ============================================
    # FIX 2: Interpretability Case Study
    # ============================================
    print("\n" + "=" * 60)
    print("FIX 2: Interpretability Comparison — EBM vs LR vs RF")
    print("=" * 60)

    from trustcert_scvul.data.ingest import create_splits
    df_case = df.copy()
    df_case = create_splits(df_case, seed=seed)

    analyzer_df_case = real_slither_to_analyzer_df(slither_feat_df, df_case, seed=seed)
    analyzer_df_case = compute_consensus_features(analyzer_df_case)

    meta_cols = ['contract_id', 'source_hash', 'vulnerability_type', 'label', 'split', 'dataset']
    full_case = df_case[meta_cols].merge(
        feat_df, on=['contract_id', 'source_hash'], how='left'
    ).merge(
        analyzer_df_case, on=['contract_id', 'source_hash'], how='left'
    ).fillna(0)

    case_studies = []

    for vuln_type in ['reentrancy', 'dos']:
        print(f"\n--- {vuln_type.upper()} Interpretability ---")
        vt_df = full_case[full_case['vulnerability_type'] == vuln_type]
        train = vt_df[vt_df['split'] == 'train']
        test = vt_df[vt_df['split'] == 'test']

        feature_cols = get_feature_columns(vt_df, vuln_type)
        X_train = np.nan_to_num(train[feature_cols].values.astype(float), nan=0.0)
        y_train = train['label'].values
        X_test = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train models
        lr = LogisticRegression(penalty='l1', solver='saga', max_iter=1000,
                                random_state=seed, class_weight='balanced')
        lr.fit(X_train_s, y_train)

        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=seed, class_weight='balanced')
        rf.fit(X_train, y_train)

        ebm = None
        if HAS_EBM:
            ebm = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                                outer_bags=8, inner_bags=4,
                                                random_state=seed)
            ebm.fit(X_train, y_train)

        # Compare feature importances
        print("\n  Top-5 features by model:")

        # LR: absolute coefficients
        lr_imp = sorted(zip(feature_cols, np.abs(lr.coef_[0])),
                       key=lambda x: x[1], reverse=True)[:5]
        print(f"  L1-Logistic (|coeff|):")
        for name, val in lr_imp:
            print(f"    {name}: {val:.4f}")

        # RF: Gini importance
        rf_imp = sorted(zip(feature_cols, rf.feature_importances_),
                       key=lambda x: x[1], reverse=True)[:5]
        print(f"  RandomForest (Gini):")
        for name, val in rf_imp:
            print(f"    {name}: {val:.4f}")

        # EBM: term importances
        if ebm:
            ebm_imp_raw = ebm.term_importances()
            ebm_imp = sorted(zip(feature_cols[:len(ebm_imp_raw)], ebm_imp_raw),
                           key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"  EBM (term importance):")
            for name, val in ebm_imp:
                print(f"    {name}: {val:.4f}")

        # Case study: TP + a contrasting (FP, FN, or abstained) case
        if ebm:
            test_contracts = test.reset_index(drop=True)
            y_pred_ebm = ebm.predict(X_test)
            y_prob_ebm = ebm.predict_proba(X_test)[:, 1]
            y_test_arr = test['label'].values

            cal_split = vt_df[vt_df['split'] == 'cal']
            X_cal = np.nan_to_num(cal_split[feature_cols].values.astype(float), nan=0.0)
            y_cal = cal_split['label'].values
            cal_probs = ebm.predict_proba(X_cal)[:, 1]
            q_hat = calibrate_conformal(y_cal, cal_probs, alpha=0.10)
            # Per-sample accept mask (singleton vs non-singleton conformal set)
            nonconf_pos = 1 - y_prob_ebm  # if we claim positive
            nonconf_neg = y_prob_ebm      # if we claim negative
            set_has_pos = nonconf_pos <= q_hat
            set_has_neg = nonconf_neg <= q_hat
            accepted_mask = set_has_pos ^ set_has_neg  # singleton iff exactly one

            def emit_case(idx, label_tag):
                contract_id = test_contracts.iloc[idx]['contract_id']
                print(f"\n  Case Study [{label_tag}]: Contract '{contract_id}' for {vuln_type}")
                local_exp = ebm.explain_local(X_test[idx:idx+1])
                feat_names = local_exp.data(0)['names']
                feat_scores = local_exp.data(0)['scores']
                feat_vals = local_exp.data(0)['values']
                top_contribs = sorted(zip(feat_names, feat_scores, feat_vals),
                                      key=lambda x: abs(x[1]), reverse=True)[:8]
                print(f"  {'Feature':<40} {'Value':>10} {'Contribution':>14}")
                print(f"  {'-'*64}")
                for fname, fscore, fval in top_contribs:
                    direction = "vuln+" if fscore > 0 else "safe-"
                    try:
                        val_str = f"{float(fval):>10.2f}"
                    except (ValueError, TypeError):
                        val_str = f"{str(fval):>10}"
                    print(f"  {fname:<40} {val_str} {fscore:>+10.4f} {direction}")
                case_studies.append({
                    'vulnerability': vuln_type,
                    'case_type': label_tag,
                    'contract_id': contract_id,
                    'y_true': int(y_test_arr[idx]),
                    'y_pred_ebm': int(y_pred_ebm[idx]),
                    'y_prob_ebm': float(y_prob_ebm[idx]),
                    'accepted_by_conformal': bool(accepted_mask[idx]),
                    'top_features': [(n, round(float(s), 4), str(v)) for n, s, v in top_contribs],
                })
                lr_pred = lr.predict(X_test_s[idx:idx+1])[0]
                rf_pred = rf.predict(X_test[idx:idx+1])[0]
                print(f"\n  Model predictions for this contract:")
                print(f"    EBM: pred={y_pred_ebm[idx]}, prob={y_prob_ebm[idx]:.3f}, "
                      f"accepted_by_conformal={bool(accepted_mask[idx])}")
                print(f"    LR:  pred={lr_pred}")
                print(f"    RF:  pred={rf_pred}")

            # Case A: TP (accepted, correct)
            tp_indices = [i for i in range(len(y_test_arr))
                          if y_test_arr[i] == 1 and y_pred_ebm[i] == 1 and accepted_mask[i]]
            if tp_indices:
                emit_case(tp_indices[0], 'TP-accepted')

            # Case B: contrast — FP, FN, or abstained (prefer abstained high-risk)
            abstained_indices = [i for i in range(len(y_test_arr)) if not accepted_mask[i]]
            fp_indices = [i for i in range(len(y_test_arr))
                          if y_test_arr[i] == 0 and y_pred_ebm[i] == 1]
            fn_indices = [i for i in range(len(y_test_arr))
                          if y_test_arr[i] == 1 and y_pred_ebm[i] == 0]

            if abstained_indices:
                # Pick an abstained contract with the probability closest to 0.5 (most uncertain)
                abstained_indices.sort(key=lambda i: abs(y_prob_ebm[i] - 0.5))
                emit_case(abstained_indices[0], 'Abstained-uncertain')
            elif fp_indices:
                emit_case(fp_indices[0], 'FP')
            elif fn_indices:
                emit_case(fn_indices[0], 'FN')

    if case_studies:
        with open(os.path.join(base_dir, 'artifacts', 'case_studies.json'), 'w') as f:
            json.dump(case_studies, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Round 3 fixes complete.")
    print("=" * 60)


if __name__ == '__main__':
    run_round3_fixes()
