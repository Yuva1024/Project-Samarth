# advanced_helpers.py
import pandas as pd
import numpy as np
# In advanced_helpers.py or a config module

PUNJAB_DISTRICTS = [
    'Gurdaspur', 'Pathankot', 'Amritsar', 'Tarn Taran', 'Kapurthala', 'Jalandhar',
    'SBS Nagar', 'Hoshiarpur', 'Rupnagar', 'SAS Nagar', 'Ludhiana', 'Ferozepur',
    'Fazilka', 'Faridkot', 'Sri Muktsar Sahib', 'Moga', 'Bathinda', 'Mansa',
    'Sangrur', 'Barnala', 'Patiala', 'Fatehgarh Sahib', 'Total'
]

KARNATAKA_DISTRICTS = [
    'Bagalkote', 'Bengaluru - Urban', 'Bengaluru - Rural', 'Belagavi', 'Ballari',
    'Bidar', 'Vijayapura', 'Chamarajanagar', 'Chickballapur', 'Chikmagalur',
    'Chitradurga', 'Dakshina Kannada', 'Davanagere', 'Dharwad', 'Gadag', 'Kalburgi',
    'Hassan', 'Haveri', 'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mysuru', 'Raichur',
    'Ramanagaram', 'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada', 'Yadgir'
]

GUJARAT_CROPS = [
    'Rice', 'Wheat', 'Jowar', 'Bajra', 'Maize', 'Ragi', 'Small Millets',
    'Total  Cereals', 'Tur (Red Gram)', 'Udad', 'Mung (Green Gram)', 'Math', 'Gram',
    'Other Pulses', 'Total  Pulses', 'Total Food Grians (A+B)'
]

CROP_TO_STATE = {
    # Gujarat crops
    'wheat': 'Gujarat',
    'rice': 'Gujarat',
    'jowar': 'Gujarat',
    'bajra': 'Gujarat',
    'maize': 'Gujarat',
    'ragi': 'Gujarat',
    'small millets': 'Gujarat',
    'total  cereals': 'Gujarat',
    'tur (red gram)': 'Gujarat',
    'udad': 'Gujarat',
    'mung (green gram)': 'Gujarat',
    'math': 'Gujarat',
    'gram': 'Gujarat',
    'other pulses': 'Gujarat',
    'total  pulses': 'Gujarat',
    'total food grians (a+b)': 'Gujarat'
}


def compare_any_metric(metric, entities, location_type, years=range(2010,2018)):
    """
    Compare rainfall, wheat, rice, or crops across states/districts/crops and years.
    :param metric: "rainfall", "wheat", "rice", crop name, etc.
    :param entities: list of state names, district names, or crop names
    :param location_type: 'state', 'district', 'crop'
    :param years: list of years
    Returns: dict with entities, years, per-entity values, and summary
    """
    if metric.lower() == 'rainfall':  # State-wise rainfall
        df = pd.read_csv('data/rainfall_subdivisional_clean.csv')
        results = {}
        for entity in entities:
            rows = df[df['subdivision'].str.contains(entity, case=False, na=False)]
            values = [rows[rows['year'] == y]['annual'].mean() for y in years]
            results[entity] = values
        header = f"**Rainfall Comparison** for: {', '.join(entities)} ({years[0]}–{years[-1]})"
    elif metric.lower() == 'wheat' and location_type == 'district':
        df = pd.read_csv('data/wheat_punjab_clean.csv')
        results = {}
        for entity in entities:
            row = df[df['district/year'].str.lower() == entity.lower()]
            if not row.empty:
                values = [float(row[str(y)]) for y in years if str(y) in row.columns]
                results[entity] = values
            else:
                results[entity] = [np.nan] * len(years)
        header = f"**Wheat Production in Punjab Districts**: {', '.join(entities)} ({years[0]}–{years[-1]})"
    elif metric.lower() == 'rice' and location_type == 'district':
        df = pd.read_csv('data/rice_karnataka_clean.csv')
        results = {}
        for entity in entities:
            row = df[df['district_name'].str.lower() == entity.lower()]
            if not row.empty:
                val = float(row['all_seasons_production'].iloc[0])
                values = [val] * len(years)
                results[entity] = values
            else:
                results[entity] = [np.nan] * len(years)
        header = f"**Rice Production (All Seasons, Karnataka Districts)**: {', '.join(entities)}"
    elif location_type == 'crop':
        df = pd.read_csv('data/crops_gujarat_clean.csv')
        results = {}
        for crop in entities:
            row = df[df['crop'].str.lower() == crop.lower()]
            if not row.empty:
                area = row['area'].iloc[0]
                production = row['production'].iloc[0]
                yield_ = row['yield'].iloc[0]
                values = [f"area={area}, production={production}, yield={yield_}"] * len(years)
                results[crop] = values
            else:
                results[crop] = ["missing"] * len(years)
        header = f"**Major Crops of Gujarat**: {', '.join(entities)}"
    else:
        return {"summary": "Requested comparison not supported."}

    lines = [header]
    for i, y in enumerate(years):
        line = f"- {y}: " + ", ".join([f"{e} = {results[e][i]}" for e in entities])
        lines.append(line)
    lines.append("\n📊 Source: Project Samarth cleaned dataset(s)")
    return {
        'entities': entities,
        'years': list(years),
        'values': results,
        'summary': '\n'.join(lines)
    }

def correlate_wheat_and_rainfall_punjab(years=range(2010,2018), district='Amritsar'):
    # As before; not modified for simplicity here
    wheat_df = pd.read_csv('data/wheat_punjab_clean.csv')
    rainfall_df = pd.read_csv('data/rainfall_subdivisional_clean.csv')
    if district in wheat_df['district/year'].values:
        row = wheat_df[wheat_df['district/year'] == district]
        wheat_prod = [float(row[str(y)]) for y in years if str(y) in row.columns]
    else:
        return {"summary": f"{district} data not found in wheat_punjab_clean.csv"}

    punjab_rain = [rainfall_df[(rainfall_df['subdivision'].str.contains('Punjab', case=False, na=False)) & (rainfall_df['year'] == y)]['annual'].mean() for y in years]
    if len(wheat_prod) == len(punjab_rain) and len(wheat_prod) > 2:
        corr = np.corrcoef(wheat_prod, punjab_rain)[0, 1]
        summary = f"""
**Wheat vs Rainfall Correlation (Amritsar, Punjab 2010-2017)**

| Year | Wheat prod (Amritsar) | Punjab rainfall (mm) |
|------|----------------------|----------------------|
""" + "\n".join([f"| {y} | {wheat_prod[i]:.1f} | {punjab_rain[i]:.1f} |" for i, y in enumerate(years)]) + f"""

**Correlation coefficient:** {corr:.2f}
Interpretation: {"Strong positive" if corr > 0.7 else "Moderate" if corr > 0.3 else "Low" if corr > 0 else "Negative"} correlation.

📊 Source: Wheat Punjab, IMD Rainfall Punjab
"""
        return {
            'years': list(years),
            'wheat_production': wheat_prod,
            'punjab_rainfall': punjab_rain,
            'correlation': corr,
            'summary': summary
        }
    return {"summary": "Insufficient data for correlation analysis."}

def handle_complex_query(prompt):
    """
    Dispatcher for advanced queries. Supports flexible rainfall/district/crop comparisons, and
    correlation analysis for wheat and rainfall in Punjab.
    """
    prompt_lower = prompt.lower()
    years = list(range(2010, 2018))
    # 1. Rainfall comparison - any N states:
    if 'compare' in prompt_lower and 'rainfall' in prompt_lower:
        # Quick-dirty state extractor
        known_states = ['Punjab', 'Karnataka', 'Gujarat', 'Maharashtra', 'Kerala', 'Tamil', 'Haryana', 'Rajasthan', 'Bihar', 'West', 'Madhya', 'Andhra', 'Chhattisgarh', 'Odisha', 'Assam']
        found = [s for s in known_states if s.lower() in prompt_lower]
        if found:
            return compare_any_metric('rainfall', found, 'state', years)
    # 2. Wheat or rice (district-wise) comparison
    elif 'compare' in prompt_lower and 'wheat' in prompt_lower and 'district' in prompt_lower:
        # Extract possible districts; enhance as needed
        known_districts = ['Amritsar','Gurdaspur','Pathankot','Ludhiana']
        found = [d for d in known_districts if d.lower() in prompt_lower]
        if found:
            return compare_any_metric('wheat', found, 'district', years)
    elif 'compare' in prompt_lower and 'rice' in prompt_lower and 'district' in prompt_lower:
        known_kar_districts = ['Bagalkote','Bengaluru - Urban','Bengaluru - Rural']
        found = [d for d in known_kar_districts if d.lower() in prompt_lower]
        if found:
            return compare_any_metric('rice', found, 'district', years)
    # 3. Crop comparison in Gujarat
    elif 'compare' in prompt_lower and 'crop' in prompt_lower and 'gujarat' in prompt_lower:
        df = pd.read_csv('data/crops_gujarat_clean.csv')
        all_crops = df['crop'].str.lower().tolist()
        found = [c for c in all_crops if c in prompt_lower]
        if found:
            return compare_any_metric(found[0], found, 'crop')
    # 4. Wheat & rainfall correlation (Amritsar)
    elif ('correlate' in prompt_lower or 'relation' in prompt_lower or 'trend' in prompt_lower) and 'wheat' in prompt_lower and 'rainfall' in prompt_lower and 'punjab' in prompt_lower:
        return correlate_wheat_and_rainfall_punjab()
    return None
