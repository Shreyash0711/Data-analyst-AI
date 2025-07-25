import pandas as pd
import polars as pl
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    try:
        df = pl.read_csv(file_path).to_pandas()
        logging.info(f"Loaded CSV with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return None