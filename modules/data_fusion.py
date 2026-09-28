import pandas as pd

def align_schemas(df_list):
    # Get common columns
    if not df_list:
        return []
    common_cols = set(df_list[0].columns)
    for df in df_list[1:]:
        common_cols.intersection_update(set(df.columns))
    
    # Return dataframes with common columns
    return [df[list(common_cols)] for df in df_list]

def fuse_data(df_list, join_type='outer'):
    if not df_list:
        return pd.DataFrame()
    if len(df_list) == 1:
        return df_list[0]
    
    # Simple concatenation for fusion if no specific key provided,
    # In a real scenario, this would align by foreign keys
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df
