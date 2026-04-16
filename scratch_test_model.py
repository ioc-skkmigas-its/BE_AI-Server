import pandas as pd
from app.core.config import get_settings
from app.infrastructure.model_loader import XGBoostModelLoader

def run_test():
    file_path = "data/trident_clean_top3000.parquet"
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return

    print(f"Data loaded, shape: {df.shape}")

    settings = get_settings()
    loader = XGBoostModelLoader(settings)
    
    print("Resolving feature columns...")
    feature_cols = loader.resolve_feature_columns(list(df.columns))
    print(f"Model requires {len(feature_cols)} features")
    
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        print(f"WARNING: The following required features are missing in the data: {missing_cols}")
        for col in missing_cols:
            df[col] = 0
            
    print("Running predictions...")
    try:
        preds = loader.predict(df[feature_cols])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    df["predicted_score"] = preds
    
    # Define columns for display and deduplication
    # We want to keep unique wells. In this dataset, WELL_NAME + FIELD_NAME is usually unique to a well.
    # Group by well identifiers and keep the record with the highest score
    id_cols = ["FID", "WELL_NAME", "FIELD_NAME"]
    
    # Deduplicate: Group by identifiers and take the maximum score
    well_groups = df.sort_values("predicted_score", ascending=False).drop_duplicates(subset=["WELL_NAME", "FIELD_NAME"])
    
    # Take top 30 unique wells
    top_30 = well_groups.head(30)
    
    # Output display
    print("\n" + "="*80)
    print(f"{'FID':<8} {'WELL_NAME':<30} {'FIELD_NAME':<20} {'predicted_score'}")
    print("-" * 80)
    
    for _, row in top_30.iterrows():
        print(f"{int(row['FID']):<8} {str(row['WELL_NAME']):<30} {str(row['FIELD_NAME']):<20} {row['predicted_score']:.2f}")
    print("="*80)
    print(f"Total unique wells identified: {len(well_groups)}")

if __name__ == "__main__":
    run_test()
