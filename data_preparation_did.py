import pandas as pd
import numpy as np

np.random.seed(42)

def prepare_did_dataset(ads_2020, ads_2024, turnout_2020, turnout_2024, demographics, dma_mapping):
    
    nostalgia_2020_dma = ads_2020.groupby('dma').agg(
        nostalgic_count=('nostalgic', 'sum'),
        total_count=('nostalgic', 'count')
    ).reset_index()
    nostalgia_2020_dma['nostalgia_pct'] = (nostalgia_2020_dma['nostalgic_count'] / nostalgia_2020_dma['total_count']) * 100
    
    nostalgia_2024_dma = ads_2024.groupby('dma').agg(
        nostalgic_count=('nostalgic', 'sum'),
        total_count=('nostalgic', 'count')
    ).reset_index()
    nostalgia_2024_dma['nostalgia_pct'] = (nostalgia_2024_dma['nostalgic_count'] / nostalgia_2024_dma['total_count']) * 100
    
    county_nostalgia_2020 = dma_mapping.merge(nostalgia_2020_dma[['dma', 'nostalgia_pct']], on='dma', how='left')
    county_nostalgia_2024 = dma_mapping.merge(nostalgia_2024_dma[['dma', 'nostalgia_pct']], on='dma', how='left')
    
    state_means_2020 = county_nostalgia_2020.groupby('state')['nostalgia_pct'].mean()
    state_means_2024 = county_nostalgia_2024.groupby('state')['nostalgia_pct'].mean()
    
    county_nostalgia_2020['nostalgia_pct'] = county_nostalgia_2020.apply(
        lambda row: state_means_2020[row['state']] if pd.isna(row['nostalgia_pct']) else row['nostalgia_pct'],
        axis=1
    )
    county_nostalgia_2024['nostalgia_pct'] = county_nostalgia_2024.apply(
        lambda row: state_means_2024[row['state']] if pd.isna(row['nostalgia_pct']) else row['nostalgia_pct'],
        axis=1
    )
    
    turnout_2020['turnout_pct'] = (turnout_2020['total_votes'] / turnout_2020['population']) * 100
    turnout_2024['turnout_pct'] = (turnout_2024['total_votes'] / turnout_2024['population']) * 100
    
    df = county_nostalgia_2020[['county_fips', 'state', 'county_name']].copy()
    df = df.merge(county_nostalgia_2020[['county_fips', 'nostalgia_pct']], on='county_fips', how='left', suffixes=('', '_2020'))
    df = df.merge(county_nostalgia_2024[['county_fips', 'nostalgia_pct']], on='county_fips', how='left', suffixes=('_2020', '_2024'))
    df = df.merge(turnout_2020[['county_fips', 'turnout_pct']], on='county_fips', how='left')
    df = df.merge(turnout_2024[['county_fips', 'turnout_pct']], on='county_fips', how='left', suffixes=('_2020', '_2024'))
    df = df.merge(demographics[['county_fips', 'pct_white', 'pct_college', 'median_income']], on='county_fips', how='left')
    
    df.rename(columns={'nostalgia_pct': 'nostalgia_2020', 'nostalgia_pct_2024': 'nostalgia_2024'}, inplace=True)
    
    df['delta_nostalgia'] = df['nostalgia_2024'] - df['nostalgia_2020']
    df['delta_turnout'] = df['turnout_2024'] - df['turnout_2020']
    
    df = df.dropna(subset=['delta_nostalgia', 'delta_turnout'])
    
    return df

def calculate_descriptive_statistics(df):
    descriptive_stats = df.groupby('state').agg({
        'nostalgia_2020': ['mean', 'std'],
        'nostalgia_2024': ['mean', 'std'],
        'delta_nostalgia': ['mean', 'std'],
        'turnout_2020': ['mean', 'std'],
        'turnout_2024': ['mean', 'std'],
        'delta_turnout': ['mean', 'std']
    }).round(2)
    return descriptive_stats

