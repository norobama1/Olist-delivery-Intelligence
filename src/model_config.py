from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "delayed"

NUMERIC_FEATURES =[
    "estimated_days",
    "approval_hours",
    "product_weight_g",
    "product_volume_cm3",
    "order_value",
    "freight_value",
    "zip_distance_proxy",
    "seller_delay_rate",
]

PASSTHROUGH_FEATURES =[
    "purchase_dayofweek",
    "purchase_month",
    "is_peak_delayed_period",
    "is_weekend", # derived in add_derived_features()
]

ORDINAL_CATEGORICAL_FEATURES =[
    "seller_state",
    "customer_state",
]
#Excluded: leakage(post-delivery), raw-timestamps

ALL_FEATURES = NUMERIC_FEATURES + PASSTHROUGH_FEATURES + ORDINAL_CATEGORICAL_FEATURES

def add_derived_features(df):
    # Requires df sorted by order_purchase_timestamp before calling.
    df = df.copy()
    df["is_weekend"] = df["purchase_dayofweek"].isin([5, 6]).astype(int)
    # shift(1) ensures only past orders from the same seller inform each row
    df["seller_delay_rate"] = (
        df.groupby("seller_id")["delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
    )
    return df

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num",numeric_pipe,NUMERIC_FEATURES),
            ("pass", "passthrough",PASSTHROUGH_FEATURES),
            ("cat",categorical_pipe,ORDINAL_CATEGORICAL_FEATURES),
        ], remainder="drop"
    )

