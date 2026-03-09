import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    
    matches = pd.merge(anon_df, aux_df, on=["age", "zip3", "gender"])

    # Gets only single 1 to 1 matches
    counts = matches.groupby("anon_id").size()
    unique_ids = counts[counts == 1].index
    unique = matches[matches["anon_id"].isin(unique_ids)]
    
    return unique[["anon_id", "name"]]


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df)/len(anon_df)