# backend/model_utils.py
"""
Helper functions to convert uploaded CSV/JSON into the numeric feature array
the model expects.

You MUST ensure the column ordering and exact features match what you trained on.
Adjust `FEATURE_COLUMNS` to match your training feature list (in the same order).
"""

import numpy as np
import pandas as pd

# IMPORTANT: set this to the exact list (and order) of feature column names used during training
# Example (fill with the final features you used):
FEATURE_COLUMNS = [
    # example features used earlier — replace with your actual 44 feature names
    'src_ip_numeric', 'src_port', 'dst_port', 'proto', 'pktTotalCount', 'octetTotalCount',
    'min_ps', 'max_ps', 'avg_ps', 'std_dev_ps', 'flowStart', 'flowEnd',
    'flowDuration', 'min_piat', 'max_piat', 'avg_piat', 'std_dev_piat',
    'f_pktTotalCount', 'f_octetTotalCount', 'f_min_ps', 'f_max_ps', 'f_avg_ps', 'f_std_dev_ps',
    'f_flowStart', 'f_flowEnd', 'f_flowDuration', 'f_min_piat', 'f_max_piat', 'f_avg_piat',
    'f_std_dev_piat', 'b_pktTotalCount', 'b_octetTotalCount', 'b_min_ps', 'b_max_ps',
    'b_avg_ps', 'b_std_dev_ps', 'b_flowStart', 'b_flowEnd', 'b_flowDuration',
    'b_min_piat', 'b_max_piat', 'b_avg_piat', 'b_std_dev_piat', 'flowEndReason'
]

def prepare_features_for_prediction(df: pd.DataFrame, scaler) -> np.ndarray:
    """
    1. Ensure all required columns present (if missing, fill with zeros)
    2. Drop non-numeric columns (like 'category', 'flow_key', 'src_ip', 'dst_ip')
    3. Reorder to FEATURE_COLUMNS order
    4. Fill NA / inf -> 0
    5. Apply scaler.transform and return numpy array
    """

    # make a copy to avoid modifying incoming df
    X = df.copy()

    # drop obviously unwanted columns if present
    drop_candidates = ['flow_key', 'src_ip', 'dst_ip', 'application_protocol', 'web_service', 'category']
    for c in drop_candidates:
        if c in X.columns:
            X = X.drop(columns=[c])

    # ensure all FEATURE_COLUMNS present
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0  # default 0 if missing

    # select and reorder features
    X = X[FEATURE_COLUMNS]

    # replace bad values
    X = X.replace([float('inf'), -float('inf')], 0)
    X = X.fillna(0)

    # convert to numeric
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

    # scale
    X_scaled = scaler.transform(X.values)

    return X_scaled
