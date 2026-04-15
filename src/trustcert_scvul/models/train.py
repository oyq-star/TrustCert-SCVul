"""Model training: baselines and TrustCert-SCVul interpretable models."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss
)
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


VULN_TYPES = ['reentrancy', 'arithmetic', 'dos', 'timestamp']


def get_feature_columns(df: pd.DataFrame, vuln_type: str) -> list:
    """Get relevant feature columns for a given vulnerability type."""
    exclude_cols = {
        'contract_id', 'source_hash', 'vulnerability_type', 'label',
        'split', 'dataset', 'source_text', 'source_path',
        'compiler_version_raw'
    }
    # Include structural features + analyzer/consensus features for this vuln type
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        # Include all structural features
        if not any(col.startswith(p) for p in ['slither_', 'mythril_', 'consensus_']):
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
            continue
        # Include analyzer/consensus features only for current vuln type or global
        if any(col.endswith(f'_{vuln_type}') or col.endswith('_timeout')
               or col.startswith('any_tool') or col.endswith('_total_findings')
               for _ in [None]):
            if vuln_type in col or 'timeout' in col or 'any_tool' in col or 'total_findings' in col:
                feature_cols.append(col)

    return feature_cols


def get_structural_only_cols(df: pd.DataFrame) -> list:
    """Get only structural feature columns (no analyzer/consensus)."""
    exclude_cols = {
        'contract_id', 'source_hash', 'vulnerability_type', 'label',
        'split', 'dataset', 'source_text', 'source_path',
        'compiler_version_raw'
    }
    return [c for c in df.columns
            if c not in exclude_cols
            and not any(c.startswith(p) for p in ['slither_', 'mythril_', 'consensus_', 'any_tool'])
            and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]


def get_analyzer_only_cols(df: pd.DataFrame, vuln_type: str) -> list:
    """Get only analyzer/consensus feature columns."""
    return [c for c in df.columns
            if any(c.startswith(p) for p in ['slither_', 'mythril_', 'consensus_', 'any_tool'])
            and (vuln_type in c or 'timeout' in c or 'any_tool' in c or 'total_findings' in c)
            and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics."""
    metrics = {}
    try:
        metrics['auprc'] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics['auprc'] = 0.0
    try:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics['auroc'] = 0.5
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    try:
        metrics['brier'] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics['brier'] = 1.0
    return metrics


def train_rule_baselines(df: pd.DataFrame, vuln_type: str) -> dict:
    """Train simple rule-based baselines (Slither-only, Mythril-only, OR, AND)."""
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    y_test = test['label'].values
    results = {}

    # Slither-only
    sl_col = f'slither_{vuln_type}_has_finding'
    if sl_col in df.columns:
        y_pred = test[sl_col].values
        y_prob = test[f'slither_{vuln_type}_max_confidence'].values / 3.0
        results['slither_only'] = compute_metrics(y_test, y_pred, y_prob)

    # Mythril-only
    my_col = f'mythril_{vuln_type}_has_finding'
    if my_col in df.columns:
        y_pred = test[my_col].values
        y_prob = test[f'mythril_{vuln_type}_max_confidence'].values / 3.0
        results['mythril_only'] = compute_metrics(y_test, y_pred, y_prob)

    # OR consensus
    if sl_col in df.columns and my_col in df.columns:
        y_pred = ((test[sl_col] | test[my_col])).astype(int).values
        y_prob = np.maximum(
            test[f'slither_{vuln_type}_max_confidence'].values,
            test[f'mythril_{vuln_type}_max_confidence'].values
        ) / 3.0
        results['or_consensus'] = compute_metrics(y_test, y_pred, y_prob)

    # AND consensus
    if sl_col in df.columns and my_col in df.columns:
        y_pred = ((test[sl_col] & test[my_col])).astype(int).values
        y_prob = np.minimum(
            test[f'slither_{vuln_type}_max_confidence'].values,
            test[f'mythril_{vuln_type}_max_confidence'].values
        ) / 3.0
        results['and_consensus'] = compute_metrics(y_test, y_pred, y_prob)

    return results


def train_ml_baselines(df: pd.DataFrame, vuln_type: str, seed=42) -> dict:
    """Train ML baselines: Logistic, LightGBM, MLP."""
    feature_cols = get_feature_columns(df, vuln_type)
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    X_train = train[feature_cols].values.astype(float)
    y_train = train['label'].values
    X_test = test[feature_cols].values.astype(float)
    y_test = test['label'].values

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    models = {}

    # L1 Logistic Regression
    lr = LogisticRegression(penalty='l1', solver='saga', max_iter=1000,
                            random_state=seed, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    y_prob = lr.predict_proba(X_test_s)[:, 1]
    y_pred = lr.predict(X_test_s)
    results['l1_logistic'] = compute_metrics(y_test, y_pred, y_prob)
    models['l1_logistic'] = (lr, scaler, feature_cols)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=seed, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    results['random_forest'] = compute_metrics(y_test, y_pred, y_prob)
    models['random_forest'] = (rf, None, feature_cols)

    # LightGBM
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=seed, class_weight='balanced',
            verbose=-1, n_jobs=-1
        )
        lgb_model.fit(X_train, y_train)
        y_prob = lgb_model.predict_proba(X_test)[:, 1]
        y_pred = lgb_model.predict(X_test)
        results['lightgbm'] = compute_metrics(y_test, y_pred, y_prob)
        models['lightgbm'] = (lgb_model, None, feature_cols)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                        random_state=seed, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    y_prob = mlp.predict_proba(X_test_s)[:, 1]
    y_pred = mlp.predict(X_test_s)
    results['mlp'] = compute_metrics(y_test, y_pred, y_prob)
    models['mlp'] = (mlp, scaler, feature_cols)

    return results, models


def train_trustcert_model(df: pd.DataFrame, vuln_type: str, seed=42) -> dict:
    """Train TrustCert-SCVul interpretable model (EBM or fallback to GBM)."""
    feature_cols = get_feature_columns(df, vuln_type)
    train = df[df['split'] == 'train']
    val = df[df['split'] == 'val']
    test = df[df['split'] == 'test']

    X_train = train[feature_cols].values.astype(float)
    y_train = train['label'].values
    X_val = val[feature_cols].values.astype(float)
    y_val = val['label'].values
    X_test = test[feature_cols].values.astype(float)
    y_test = test['label'].values

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    results = {}
    model_info = {}

    if HAS_EBM:
        # Explainable Boosting Machine (InterpretML)
        ebm = ExplainableBoostingClassifier(
            max_bins=64,
            interactions=5,
            outer_bags=8,
            inner_bags=4,
            random_state=seed,
        )
        ebm.fit(X_train, y_train)
        y_prob = ebm.predict_proba(X_test)[:, 1]
        y_pred = ebm.predict(X_test)
        results['trustcert_ebm'] = compute_metrics(y_test, y_pred, y_prob)
        model_info['trustcert_ebm'] = {
            'model': ebm,
            'feature_names': feature_cols,
            'type': 'EBM',
        }

        # Get feature importances
        importances = ebm.term_importances()
        top_features = sorted(
            zip(feature_cols[:len(importances)], importances),
            key=lambda x: abs(x[1]), reverse=True
        )[:10]
        model_info['trustcert_ebm']['top_features'] = top_features
    else:
        # Fallback: Gradient Boosting with shallow trees (more interpretable)
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=seed
        )
        gbm.fit(X_train, y_train)
        y_prob = gbm.predict_proba(X_test)[:, 1]
        y_pred = gbm.predict(X_test)
        results['trustcert_gbm'] = compute_metrics(y_test, y_pred, y_prob)

        top_features = sorted(
            zip(feature_cols, gbm.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:10]
        model_info['trustcert_gbm'] = {
            'model': gbm,
            'feature_names': feature_cols,
            'type': 'GBM_shallow',
            'top_features': top_features,
        }

    # Ablation: structural-only
    struct_cols = get_structural_only_cols(df)
    X_train_s = train[struct_cols].values.astype(float)
    X_test_s = test[struct_cols].values.astype(float)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0)

    if HAS_EBM:
        ebm_s = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                              random_state=seed)
        ebm_s.fit(X_train_s, y_train)
        y_prob = ebm_s.predict_proba(X_test_s)[:, 1]
        y_pred = ebm_s.predict(X_test_s)
        results['structural_only'] = compute_metrics(y_test, y_pred, y_prob)
    else:
        gbm_s = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           random_state=seed)
        gbm_s.fit(X_train_s, y_train)
        y_prob = gbm_s.predict_proba(X_test_s)[:, 1]
        y_pred = gbm_s.predict(X_test_s)
        results['structural_only'] = compute_metrics(y_test, y_pred, y_prob)

    # Ablation: analyzer-only
    ana_cols = get_analyzer_only_cols(df, vuln_type)
    if ana_cols:
        X_train_a = train[ana_cols].values.astype(float)
        X_test_a = test[ana_cols].values.astype(float)
        X_train_a = np.nan_to_num(X_train_a, nan=0.0)
        X_test_a = np.nan_to_num(X_test_a, nan=0.0)

        if HAS_EBM:
            ebm_a = ExplainableBoostingClassifier(max_bins=64, interactions=5,
                                                  random_state=seed)
            ebm_a.fit(X_train_a, y_train)
            y_prob = ebm_a.predict_proba(X_test_a)[:, 1]
            y_pred = ebm_a.predict(X_test_a)
            results['analyzer_only'] = compute_metrics(y_test, y_pred, y_prob)
        else:
            gbm_a = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               random_state=seed)
            gbm_a.fit(X_train_a, y_train)
            y_prob = gbm_a.predict_proba(X_test_a)[:, 1]
            y_pred = gbm_a.predict(X_test_a)
            results['analyzer_only'] = compute_metrics(y_test, y_pred, y_prob)

    return results, model_info


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, alpha=0.05, seed=42):
    """Compute bootstrap 95% confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_boot = {'f1': [], 'precision': [], 'recall': [], 'auprc': []}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]

        if len(np.unique(yt)) < 2:
            continue

        metrics_boot['f1'].append(f1_score(yt, yp, zero_division=0))
        metrics_boot['precision'].append(precision_score(yt, yp, zero_division=0))
        metrics_boot['recall'].append(recall_score(yt, yp, zero_division=0))
        try:
            metrics_boot['auprc'].append(average_precision_score(yt, ypr))
        except Exception:
            pass

    ci = {}
    for metric, values in metrics_boot.items():
        if len(values) < 10:
            ci[metric] = (0.0, 0.0)
            continue
        lo = np.percentile(values, 100 * alpha / 2)
        hi = np.percentile(values, 100 * (1 - alpha / 2))
        ci[metric] = (round(lo, 3), round(hi, 3))
    return ci
