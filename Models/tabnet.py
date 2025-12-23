import numpy as np
import pandas as pd
import kagglehub
import wandb
import torch

from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
)
from pytorch_tabnet.tab_model import TabNetClassifier

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
data_path = path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

useless_cols = [
    "customerID",
    "MultipleLines",
]

rename_map = {
    "tenure": "subscription_tenure_months",
    "Contract": "billing_cycle_type",
    "MonthlyCharges": "MonthlyCharges",
    "TotalCharges": "TotalCharges",
    "InternetService": "PlanType",
    "PhoneService": "ProductEnabled",
    "OnlineBackup": "OnlineBackup",
    "StreamingTV": "FeatureA",
    "StreamingMovies": "FeatureB",
    "PaymentMethod": "PaymentMethod",
    "SeniorCitizen": "is_senior_user",
    "Partner": "has_partner",
    "Dependents": "has_dependents",
}

existing = [c for c in useless_cols if c in df.columns]
df = df.drop(columns=existing)
rename_map_existing = {k: v for k, v in rename_map.items() if k in df.columns}
df = df.rename(columns=rename_map_existing)

# fix TotalCharges as usual (should be numbers!)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


target_le = LabelEncoder()
df["Churn"] = target_le.fit_transform(df["Churn"])

X_all = df.drop(columns=["Churn"])
y_all = df["Churn"]
X_all = pd.get_dummies(X_all, drop_first=True)


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

scaler = StandardScaler()
X_train_full = pd.DataFrame(
    scaler.fit_transform(X_train_full),
    columns=X_train_full.columns,
    index=X_train_full.index,
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

feature_cols = X_train_full.columns.tolist()

X_train_full_tab = X_train_full[feature_cols].values.astype(np.float32)
X_test_tab = X_test[feature_cols].values.astype(np.float32)
y_train_full_np = y_train_full.values
y_test_np = y_test.values

pos = y_train_full_np.sum()
neg = (y_train_full_np == 0).sum()
spw = neg / pos
class_weights_tensor = torch.tensor([1.0, float(spw)], dtype=torch.float32)

project_name = "telco-churn-tabnet-cv-sweep"


def main():
    # W&B Sweeps setup
    default_config = dict(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.3,
        lambda_sparse=1e-4,
        lr=2e-2,
        max_epochs=200,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
    )

    run = wandb.init(project=project_name, config=default_config)
    cfg = wandb.config

    # run can;t be empty
    wandb.log({"run_started": 1})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = []
    best_ts = []
    fold_f1s = []
    fold_recalls = []
    fold_aucs = []
    fold_accs = []

    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

    fold_idx = 1
    for train_idx, val_idx in cv.split(X_train_full_tab, y_train_full_np):
        X_tr = X_train_full_tab[train_idx]
        y_tr = y_train_full_np[train_idx]
        X_v = X_train_full_tab[val_idx]
        y_v = y_train_full_np[val_idx]

        model = TabNetClassifier(
            n_d=cfg.n_d,
            n_a=cfg.n_a,
            n_steps=cfg.n_steps,
            gamma=cfg.gamma,
            lambda_sparse=cfg.lambda_sparse,
            cat_idxs=[],
            cat_dims=[],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=cfg.lr),
            verbose=0,
            mask_type="sparsemax",
            scheduler_params=None,
            scheduler_fn=None,
            seed=42,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_v, y_v)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            batch_size=cfg.batch_size,
            virtual_batch_size=cfg.virtual_batch_size,
            num_workers=0,
            drop_last=False,
            loss_fn=loss_fn,
        )

        y_val_proba = model.predict_proba(X_v)[:, 1]

        thresholds = np.linspace(0.01, 1.0, 100)
        f1_scores = []
        for t in thresholds:
            y_val_pred_t = (y_val_proba >= t).astype(int)
            f1_scores.append(f1_score(y_v, y_val_pred_t))

        best_t = thresholds[np.argmax(f1_scores)]
        best_ts.append(best_t)
        models.append(model)

        y_v_pred = (y_val_proba >= best_t).astype(int)

        roc = roc_auc_score(y_v, y_val_proba)
        rec = recall_score(y_v, y_v_pred)
        f1 = f1_score(y_v, y_v_pred)
        acc = accuracy_score(y_v, y_v_pred)
        cm = confusion_matrix(y_v, y_v_pred)

        fold_f1s.append(f1)
        fold_recalls.append(rec)
        fold_aucs.append(roc)
        fold_accs.append(acc)

        wandb.log(
            {
                "fold": fold_idx,
                "fold_best_threshold": float(best_t),
                "fold_val_roc_auc": float(roc),
                "fold_val_f1": float(f1),
                "fold_val_recall": float(rec),
                "fold_val_accuracy": float(acc),
                "fold_tn": int(cm[0, 0]),
                "fold_fp": int(cm[0, 1]),
                "fold_fn": int(cm[1, 0]),
                "fold_tp": int(cm[1, 1]),
            }
        )

        fold_idx += 1

    # ensemblee on test
    test_probas_list = [m.predict_proba(X_test_tab)[:, 1] for m in models]
    test_probas = np.column_stack(test_probas_list)
    y_test_proba = test_probas.mean(axis=1)

    global_t = float(np.mean(best_ts))
    y_test_pred = (y_test_proba >= global_t).astype(int)

    test_roc = roc_auc_score(y_test_np, y_test_proba)
    test_rec = recall_score(y_test_np, y_test_pred)
    test_f1 = f1_score(y_test_np, y_test_pred)
    test_acc = accuracy_score(y_test_np, y_test_pred)
    test_cm = confusion_matrix(y_test_np, y_test_pred)

    cv_mean_f1 = float(np.mean(fold_f1s))
    cv_mean_rec = float(np.mean(fold_recalls))
    cv_mean_roc = float(np.mean(fold_aucs))
    cv_mean_acc = float(np.mean(fold_accs))

    wandb.log(
        {
            "cv_mean_val_f1": cv_mean_f1,
            "cv_mean_val_recall": cv_mean_rec,
            "cv_mean_val_roc_auc": cv_mean_roc,
            "cv_mean_val_accuracy": cv_mean_acc,
            "cv_avg_best_threshold": global_t,
            "test_roc_auc": float(test_roc),
            "test_f1": float(test_f1),
            "test_recall": float(test_rec),
            "test_accuracy": float(test_acc),
            "test_tn": int(test_cm[0, 0]),
            "test_fp": int(test_cm[0, 1]),
            "test_fn": int(test_cm[1, 0]),
            "test_tp": int(test_cm[1, 1]),
        }
    )

    print("=== TABNET TEST REPORT (current sweep config) ====")
    print("ROC AUC:", test_roc)
    print("F1:", test_f1)
    print("Recall:", test_rec)
    print("Accuracy:", test_acc)
    print("Confusion matrix:\n", test_cm)
    print("CV mean F1:", cv_mean_f1)
    print("CV mean Recall:", cv_mean_rec)

    run.finish()


if __name__ == "__main__":
    main()
