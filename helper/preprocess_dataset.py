import pandas as pd
from sklearn.model_selection import train_test_split


def convert_number(v):
    """Convert a string value from a config file into int/float when possible."""
    v = v.strip()
    try:
        if '.' in v or 'e' in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        if v.lower() == 'none':
            return None
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        return v


def split_dataset(df, test_size=0.2, random_state=42):
    """Split a full dataset into train/test sets."""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def encode_categorical(train_df, test_df, categorical_cols):
    """
    One-hot encode categorical columns using train categories as the reference,
    so train/test end up with identical dummy columns (fit-on-train, apply-to-test).
    """
    train_dummies = pd.get_dummies(train_df[categorical_cols], prefix=categorical_cols)
    test_dummies = pd.get_dummies(test_df[categorical_cols], prefix=categorical_cols)

    # align test columns to train columns (missing -> 0, extra -> dropped)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    return train_dummies.astype(int), test_dummies.astype(int)
