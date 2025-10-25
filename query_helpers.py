

import pandas as pd
import os

def get_highest_wheat_punjab(year):
    """Find district with highest wheat production in Punjab for given year"""
    try:
        df = pd.read_csv('data/wheat_punjab_clean.csv')
        year_str = str(year)
        
        if year_str not in df.columns:
            return None, None
        
        max_value = -float('inf')
        best_district = None
        
        for idx, row in df.iterrows():
            try:
                val = float(row[year_str])
                if val > max_value:
                    max_value = val
                    best_district = row['district/year']
            except:
                continue
        
        return best_district, max_value
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_wheat_production_multi_year(district, years=None):
    """Get wheat production for a district over multiple years"""
    try:
        df = pd.read_csv('data/wheat_punjab_clean.csv')
        
        # Find district row (case-insensitive)
        district_row = df[df['district/year'].str.lower() == district.lower()]
        
        if district_row.empty:
            return None
        
        if years is None:
            years = [str(y) for y in range(2022, 2017, -1)]  # Last 5 years
        
        result = {}
        for year in years:
            if str(year) in district_row.columns:
                result[year] = district_row.iloc[0][str(year)]
        
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_highest_crop_gujarat(metric='production'):
    """Find crop with highest production/yield/area in Gujarat"""
    try:
        df = pd.read_csv('data/crops_gujarat_clean.csv')
        
        if metric not in df.columns:
            return None, None
        
        idx = df[metric].idxmax()
        crop = df.loc[idx, 'crop']
        value = df.loc[idx, metric]
        
        return crop, value
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_highest_rice_karnataka(metric='all_seasons_production'):
    """Find district with highest rice production/yield in Karnataka"""
    try:
        df = pd.read_csv('data/rice_karnataka_clean.csv')
        
        if metric not in df.columns:
            return None, None
        
        idx = df[metric].idxmax()
        district = df.loc[idx, 'district_name']
        value = df.loc[idx, metric]
        
        return district, value
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_annual_rainfall_subdivision(subdivision, year):
    """Get annual rainfall for a specific subdivision and year"""
    try:
        df = pd.read_csv('data/rainfall_subdivisional_clean.csv')
        
        # Filter by subdivision and year
        result = df[(df['subdivision'].str.lower() == subdivision.lower()) & 
                    (df['year'] == int(year))]
        
        if result.empty:
            return None
        
        return result.iloc[0]['annual']
    except Exception as e:
        print(f"Error: {e}")
        return None

def detect_query_type(question):
    """Detect if question needs code-based aggregate query"""
    question_lower = question.lower()
    
    keywords_aggregate = ['highest', 'lowest', 'maximum', 'minimum', 'top', 'bottom', 
                          'most', 'least', 'best', 'worst', 'compare']
    
    keywords_timeseries = ['last 5 years', 'past 5 years', 'trend', 'over time']
    
    for keyword in keywords_aggregate:
        if keyword in question_lower:
            return 'aggregate'
    
    for keyword in keywords_timeseries:
        if keyword in question_lower:
            return 'timeseries'
    
    return 'lookup'

def list_crops_gujarat():
    """List all crops in Gujarat"""
    try:
        df = pd.read_csv('data/crops_gujarat_clean.csv')
        crops = df['crop'].tolist()
        return crops
    except Exception as e:
        print(f"Error: {e}")
        return None

def compare_wheat_punjab_gujarat(year):
    """Compare wheat production between Punjab (total) and Gujarat"""
    try:
        # Punjab data
        punjab_df = pd.read_csv('data/wheat_punjab_clean.csv')
        year_str = str(year)
        
        if year_str in punjab_df.columns:
            punjab_total = punjab_df[year_str].sum()
        else:
            punjab_total = None
        
        # Gujarat data
        gujarat_df = pd.read_csv('data/crops_gujarat_clean.csv')
        gujarat_wheat = gujarat_df[gujarat_df['crop'].str.lower() == 'wheat']
        
        if not gujarat_wheat.empty:
            gujarat_production = gujarat_wheat.iloc[0]['production']
        else:
            gujarat_production = None
        
        return punjab_total, gujarat_production
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_average_rainfall_subdivision(subdivision, start_year=2010, end_year=2017):
    """Calculate average annual rainfall for a subdivision"""
    try:
        df = pd.read_csv('data/rainfall_subdivisional_clean.csv')
        
        filtered = df[(df['subdivision'].str.lower() == subdivision.lower()) & 
                     (df['year'] >= start_year) & 
                     (df['year'] <= end_year)]
        
        if filtered.empty:
            return None
        
        avg_rainfall = filtered['annual'].mean()
        return avg_rainfall, len(filtered)
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_rainfall_trend_subdivision(subdivision, start_year=2010, end_year=2017):
    """Get rainfall trend for subdivision over years"""
    try:
        df = pd.read_csv('data/rainfall_subdivisional_clean.csv')
        
        filtered = df[(df['subdivision'].str.lower() == subdivision.lower()) & 
                     (df['year'] >= start_year) & 
                     (df['year'] <= end_year)]
        
        if filtered.empty:
            return None
        
        trend_data = filtered[['year', 'annual']].sort_values('year')
        return trend_data
    except Exception as e:
        print(f"Error: {e}")
        return None
