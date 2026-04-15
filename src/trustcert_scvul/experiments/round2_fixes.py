"""Round 2 fixes: abstention baselines, structural-only Wild eval, AURC for all models."""
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trustcert_scvul.data.ingest import (
    load_smartbugs_curated, deduplicate, create_splits,
    expand_with_negatives, TARGET_VULNS
)
from trustcert_scvul.features.structural import extract_features_batch
from trustcert_scvul.features.analyzer import (
    real_slither_to_analyzer_df, compute_consensus_features
)
from trustcert_scvul.models.train import (
    get_feature_columns, get_structural_only_cols, get_analyzer_only_cols,
    train_ml_baselines, train_trustcert_model, train_rule_baselines
)
from trustcert_scvul.calibration.conformal import (
    calibrate_conformal, evaluate_selective,
    compute_risk_coverage_curve, compute_aurc
)
from trustcert_scvul.analyzers.slither_runner import (
    run_slither_batch, findings_to_features, _load_version_map
)
from trustcert_scvul.data.wild_loader import sample_wild_contracts, label_with_slither
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
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


def run_round2_fixes(base_dir=None, seed=42):
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent.parent.parent)

    raw_dir = os.path.join(base_dir, 'data', 'raw')
    curated_dir = os.path.join(raw_dir, 'smartbugs_curated', 'dataset')

    # Load data
    print("=" * 60)
    print("Round 2 Fixes: Loading data...")
    df = load_smartbugs_curated(os.path.join(raw_dir, 'smartbugs_curated'))
    df = expand_with_negatives(df)
    df = deduplicate(df)
    df = create_splits(df, seed=seed)

    unique_contracts = df.drop_duplicates(subset=['contract_id', 'source_hash'])
    feat_df = extract_features_batch(unique_contracts)

    # Real Slither
    version_map = _load_version_map(curated_dir)
    print("\nRunning Slither on curated contracts...")
    slither_findings = run_slither_batch(unique_contracts, data_dir=curated_dir,
                                         version_map=version_map)
    slither_feat_df = findings_to_features(slither_findings, unique_contracts)
    analyzer_df = real_slither_to_analyzer_df(slither_feat_df, df, seed=seed)
    analyzer_df = compute_consensus_features(analyzer_df)

    meta_cols = ['contract_id', 'source_hash', 'vulnerability_type', 'label', 'split', 'dataset']
    full_df = df[meta_cols].merge(
        feat_df, on=['contract_id', 'source_hash'], how='left'
    ).merge(
        analyzer_df, on=['contract_id', 'source_hash'], how='left'
    ).fillna(0)

    # ============================================
    # FIX 2: Abstention baselines at matched coverage
    # ============================================
    print("\n" + "=" * 60)
    print("FIX 2: Abstention Baselines at Matched Coverage")
    print("=" * 60)

    abstention_results = []

    for vuln_type in ['reentrancy', 'dos']:
        print(f"\n--- {vuln_type.upper()} ---")
        vt_df = full_df[full_df['vulnerability_type'] == vuln_type].copy()
        train = vt_df[vt_df['split'] == 'train']
        cal = vt_df[vt_df['split'] == 'cal']
        test = vt_df[vt_df['split'] == 'test']

        feature_cols = get_feature_columns(vt_df, vuln_type)
        struct_cols = get_structural_only_cols(vt_df)

        X_train = train[feature_cols].values.astype(float)
        y_train = train['label'].values
        X_cal = cal[feature_cols].values.astype(float)
        y_cal = cal['label'].values
        X_test = test[feature_cols].values.astype(float)
        y_test = test['label'].values

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_cal = np.nan_to_num(X_cal, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_test_s = scaler.transform(X_test)

        models_to_test = {}

        # L1 Logistic
        lr = LogisticRegression(penalty='l1', solver='saga', max_iter=1000,
                                random_state=seed, class_weight='balanced')
        lr.fit(X_train_s, y_train)
        models_to_test['L1-Logistic'] = (lr, X_cal_s, X_test_s)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=seed, class_weight='balanced')
        rf.fit(X_train, y_train)
        models_to_test['RandomForest'] = (rf, X_cal, X_test)

        # LightGBM
        if HAS_LGB:
            lgb_m = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                       random_state=seed, class_weight='balanced', verbose=-1)
            lgb_m.fit(X_train, y_train)
            models_to_test['LightGBM'] = (lgb_m, X_cal, X_test)

        # TrustCert EBM
        if HAS_EBM:
            ebm = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                                outer_bags=8, inner_bags=4,
                                                random_state=seed)
            ebm.fit(X_train, y_train)
            models_to_test['TrustCert-EBM'] = (ebm, X_cal, X_test)

        for alpha in [0.05, 0.10, 0.15, 0.20]:
            for model_name, (model, Xc, Xt) in models_to_test.items():
                cal_probs = model.predict_proba(Xc)[:, 1]
                test_probs = model.predict_proba(Xt)[:, 1]
                q_hat = calibrate_conformal(y_cal, cal_probs, alpha=alpha)
                sel = evaluate_selective(y_test, test_probs, q_hat, model_name=model_name)

                # Risk-coverage curve
                coverages, risks = compute_risk_coverage_curve(y_test, test_probs)
                aurc = compute_aurc(coverages, risks)

                res = {
                    'vulnerability': vuln_type,
                    'model': model_name,
                    'alpha': alpha,
                    'acceptance_rate': sel['acceptance_rate'],
                    'accepted_f1': sel['accepted_f1'],
                    'accepted_precision': sel['accepted_precision'],
                    'coverage': sel['empirical_coverage'],
                    'all_sample_f1': sel['all_sample_f1'],
                    'aurc': aurc,
                }
                abstention_results.append(res)

                if alpha == 0.10:
                    print(f"  {model_name} (alpha={alpha}): accept={sel['acceptance_rate']:.1%}, "
                          f"acc_F1={sel['accepted_f1']:.3f}, acc_P={sel['accepted_precision']:.3f}, "
                          f"AURC={aurc:.4f}")

    abstention_df = pd.DataFrame(abstention_results)
    abstention_df.to_csv(os.path.join(base_dir, 'artifacts', 'abstention_baselines.csv'), index=False)
    print(f"\nSaved to artifacts/abstention_baselines.csv")

    # Print comparison table for alpha=0.10
    print("\n=== Abstention Comparison (alpha=0.10) ===")
    for vuln_type in ['reentrancy', 'dos']:
        print(f"\n  {vuln_type.upper()}:")
        subset = abstention_df[(abstention_df['vulnerability'] == vuln_type) &
                               (abstention_df['alpha'] == 0.10)]
        print(f"  {'Model':<15} {'Accept%':>8} {'Acc.F1':>7} {'Acc.P':>7} {'AURC':>7}")
        for _, row in subset.iterrows():
            print(f"  {row['model']:<15} {row['acceptance_rate']:>7.1%} "
                  f"{row['accepted_f1']:>7.3f} {row['accepted_precision']:>7.3f} "
                  f"{row['aurc']:>7.4f}")

    # ============================================
    # FIX 1: Structural-only Wild evaluation
    # ============================================
    print("\n" + "=" * 60)
    print("FIX 1: Structural-Only Wild Evaluation (no Slither features)")
    print("=" * 60)

    wild_dir = os.path.join(raw_dir, 'smartbugs_wild')
    wild_df = sample_wild_contracts(wild_dir, n_sample=200, seed=seed)

    if len(wild_df) > 0:
        print("Running Slither on Wild for labeling only...")
        wild_findings = run_slither_batch(wild_df)
        wild_labeled = label_with_slither(wild_df, wild_findings)
        wild_feat = extract_features_batch(wild_df)

        print("\n--- Structural-only model on Wild (no Slither circularity) ---")
        struct_wild_results = []

        for vuln_type in ['reentrancy', 'dos']:
            vt_df = full_df[full_df['vulnerability_type'] == vuln_type]
            train = vt_df[vt_df['split'] == 'train']
            struct_cols = get_structural_only_cols(vt_df)

            X_train_struct = train[struct_cols].values.astype(float)
            y_train = train['label'].values
            X_train_struct = np.nan_to_num(X_train_struct, nan=0.0)

            if HAS_EBM:
                ebm_s = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                                       random_state=seed)
                ebm_s.fit(X_train_struct, y_train)
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                ebm_s = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                    random_state=seed)
                ebm_s.fit(X_train_struct, y_train)

            vt_wild = wild_labeled[wild_labeled['vulnerability_type'] == vuln_type]
            pos_count = (vt_wild['label'] == 1).sum()
            if pos_count < 3:
                print(f"  [SKIP] {vuln_type}: only {pos_count} positives")
                continue

            wild_merged = vt_wild.merge(wild_feat, on=['contract_id', 'source_hash'], how='left').fillna(0)
            available = [c for c in struct_cols if c in wild_merged.columns]
            missing = [c for c in struct_cols if c not in wild_merged.columns]
            X_wild = wild_merged[available].values.astype(float)
            if missing:
                X_wild = np.column_stack([X_wild, np.zeros((len(wild_merged), len(missing)))])
            X_wild = np.nan_to_num(X_wild, nan=0.0)
            y_wild = vt_wild['label'].values

            y_prob = ebm_s.predict_proba(X_wild)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            res = {
                'vulnerability': vuln_type,
                'model': 'Structural-only EBM',
                'n_samples': len(y_wild),
                'n_positive': int(pos_count),
                'f1': f1_score(y_wild, y_pred, zero_division=0),
                'precision': precision_score(y_wild, y_pred, zero_division=0),
                'recall': recall_score(y_wild, y_pred, zero_division=0),
            }
            struct_wild_results.append(res)
            print(f"  Structural-only Wild {vuln_type}: n={res['n_samples']}, "
                  f"pos={res['n_positive']}, F1={res['f1']:.3f}, "
                  f"P={res['precision']:.3f}, R={res['recall']:.3f}")

        if struct_wild_results:
            pd.DataFrame(struct_wild_results).to_csv(
                os.path.join(base_dir, 'artifacts', 'wild_structural_only.csv'), index=False)

    # ============================================
    # FIX 4: TP/FP/FN counts
    # ============================================
    print("\n" + "=" * 60)
    print("FIX 4: TP/FP/FN Counts for TrustCert-EBM")
    print("=" * 60)

    for vuln_type in ['reentrancy', 'dos']:
        vt_df = full_df[full_df['vulnerability_type'] == vuln_type]
        test = vt_df[vt_df['split'] == 'test']
        feature_cols = get_feature_columns(vt_df, vuln_type)

        X_train = vt_df[vt_df['split'] == 'train'][feature_cols].values.astype(float)
        y_train = vt_df[vt_df['split'] == 'train']['label'].values
        X_test = test[feature_cols].values.astype(float)
        y_test = test['label'].values
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        if HAS_EBM:
            ebm = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                                outer_bags=8, inner_bags=4,
                                                random_state=seed)
            ebm.fit(X_train, y_train)
            y_pred = ebm.predict(X_test)

            tp = int(((y_pred == 1) & (y_test == 1)).sum())
            fp = int(((y_pred == 1) & (y_test == 0)).sum())
            fn = int(((y_pred == 0) & (y_test == 1)).sum())
            tn = int(((y_pred == 0) & (y_test == 0)).sum())
            print(f"  {vuln_type}: TP={tp}, FP={fp}, FN={fn}, TN={tn} "
                  f"(total={tp+fp+fn+tn}, pos={tp+fn}, neg={fp+tn})")

    print("\n" + "=" * 60)
    print("Round 2 fixes complete.")
    print("=" * 60)


if __name__ == '__main__':
    run_round2_fixes()
