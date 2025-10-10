import pandas as pd
import re

def parse_duration_to_minutes(duration_str):
    if pd.isna(duration_str):
        return None
    hours_match = re.search(r'(\d+)H', str(duration_str))
    minutes_match = re.search(r'(\d+)M', str(duration_str))
    hours = int(hours_match.group(1)) if hours_match else 0
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    return hours * 60 + minutes

def categorize_stops(airline_string):
    if pd.isna(airline_string):
        return None
    segments = len(str(airline_string).split('||'))
    return segments - 1
