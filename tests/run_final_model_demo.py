from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.lightgbm_model import build_lightgbm_pipeline
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.data.split_data import make_train_test_split
from src.features.feature_engineering import make_feature_engineering
from src.modeling.compare import evaluate_binary_classifier
from src.modeling.train import train_model

CONFIG_DIR = PROJECT_ROOT / 'src' / 'config'


def load_yaml(path: Path) -> dict:
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def prepare_final_dataset(
    project_config: dict,
    feature_config: dict,
    final_model_config: dict,
) -> dict:
    data_path = PROJECT_ROOT / project_config['data']['central_file']
    target_column = feature_config['target']
    feature_set_name = final_model_config['feature_set']
    feature_set = feature_config['feature_sets'][feature_set_name]

    df = pd.read_csv(data_path)
    df = df.drop(columns=feature_config['drop_columns'], errors='ignore').copy()
    df = make_feature_engineering(df)

    num_features = filter_existing_features(
        feature_set.get('num', []),
        df.columns.tolist(),
    )
    cat_features = filter_existing_features(
        feature_set.get('cat', []),
        df.columns.tolist(),
    )
    feature_columns = num_features + cat_features

    X_train, X_test, y_train, y_test = make_train_test_split(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        test_size=float(final_model_config['test_size']),
        random_state=int(final_model_config['random_state']),
        stratify=True,
    )

    preprocessor = build_preprocessor(
        num_features=num_features,
        cat_features=cat_features,
        X_reference=X_train,
    )

    return {
        'data_path': data_path,
        'feature_set_name': feature_set_name,
        'num_features': num_features,
        'cat_features': cat_features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
    }


def build_final_model(final_model_config: dict, preprocessor):
    model_key = final_model_config['model_key']
    if model_key != 'lightgbm':
        raise NotImplementedError(
            f"This demo script currently supports only 'lightgbm', got: {model_key}"
        )

    params = final_model_config['params']
    return build_lightgbm_pipeline(
        preprocessor=preprocessor,
        random_state=int(final_model_config['random_state']),
        sampling_method=final_model_config['sampling_method'],
        n_estimators=int(params['n_estimators']),
        learning_rate=float(params['learning_rate']),
        num_leaves=int(params['num_leaves']),
        max_depth=int(params['max_depth']),
        min_child_samples=int(params['min_child_samples']),
        subsample=float(params['subsample']),
        colsample_bytree=float(params['colsample_bytree']),
        reg_alpha=float(params['reg_alpha']),
        reg_lambda=float(params['reg_lambda']),
        class_weight=params['class_weight'],
        n_jobs=-1,
    )


def fmt_metric(value) -> str:
    if value is None:
        return 'None'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def print_metrics_table(title: str, metrics: dict) -> None:
    ordered_items = [
        ('precision', metrics.get('precision_1')),
        ('recall', metrics.get('recall_1')),
        ('f1', metrics.get('f1_1')),
        ('average_precision', metrics.get('prc_auc')),
        ('tn', metrics.get('tn')),
        ('fp', metrics.get('fp')),
        ('fn', metrics.get('fn')),
        ('tp', metrics.get('tp')),
    ]

    print(title)
    for key, value in ordered_items:
        print(f'  - {key:<18} {fmt_metric(value)}')
    print()


def print_expected_table(expected_metrics: dict) -> None:
    ordered_items = [
        ('precision', expected_metrics.get('precision_1')),
        ('recall', expected_metrics.get('recall_1')),
        ('f1', expected_metrics.get('f1_1')),
        ('average_precision', expected_metrics.get('average_precision')),
        ('tn', expected_metrics.get('tn')),
        ('fp', expected_metrics.get('fp')),
        ('fn', expected_metrics.get('fn')),
        ('tp', expected_metrics.get('tp')),
    ]

    print('Expected metrics from model_config.yaml')
    for key, value in ordered_items:
        print(f'  - {key:<18} {fmt_metric(value)}')
    print()


def main() -> None:
    project_config = load_yaml(CONFIG_DIR / 'config.yaml')
    feature_config = load_yaml(CONFIG_DIR / 'feature.yaml')
    model_config = load_yaml(CONFIG_DIR / 'model_config.yaml')
    final_model_config = model_config['final_model']

    prepared = prepare_final_dataset(
        project_config=project_config,
        feature_config=feature_config,
        final_model_config=final_model_config,
    )

    model = build_final_model(
        final_model_config=final_model_config,
        preprocessor=prepared['preprocessor'],
    )
    trained_model = train_model(model, prepared['X_train'], prepared['y_train'])

    threshold = float(final_model_config['threshold'])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='X does not have valid feature names, but LGBMClassifier was fitted with feature names',
            category=UserWarning,
        )
        observed_metrics = evaluate_binary_classifier(
            model=trained_model,
            X_eval=prepared['X_test'],
            y_eval=prepared['y_test'],
            threshold=threshold,
        )

    positives = int(prepared['y_test'].sum())
    detected_departures = int(observed_metrics['tp'])

    print()
    print('=== FINAL MODEL DEMO ===')
    print(f"Data file        : {prepared['data_path']}")
    print(f"Model            : {final_model_config['display_name']}")
    print(f"Feature set      : {prepared['feature_set_name']}")
    print(f"Sampling         : {final_model_config['sampling_method']}")
    print(f"Threshold        : {threshold:.6f}")
    print(f"Train rows       : {len(prepared['X_train'])}")
    print(f"Test rows        : {len(prepared['X_test'])}")
    print(f"Num features     : {len(prepared['num_features'])}")
    print(f"Cat features     : {len(prepared['cat_features'])}")
    print()

    print_metrics_table('Observed metrics on the test split', observed_metrics)
    print_expected_table(final_model_config.get('test_metrics', {}))

    print('Business reading')
    print(
        '  - Departures detected : '
        f"{detected_departures}/{positives} ({detected_departures / positives:.1%})"
    )
    print(
        '  - Main trade-off      : fewer false negatives, '
        'more preventive alerts'
    )
    print()


if __name__ == '__main__':
    main()
