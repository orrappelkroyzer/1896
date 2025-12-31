import pandas as pd
import sys
import re
import time
import requests
from pathlib import Path
local_python_path = str(Path(__file__).parents[0])
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
from utils.plotly_utils import write_csv, fix_and_write
import plotly.express as px
from datetime import datetime
logger = get_logger(__name__)
config = load_config(config_path=Path(local_python_path)/ 'config.json', add_date=False)


def split_dataset_to_csvs(dataset):
    ds = dataset["1896"]  # the Dataset inside the DatasetDict

    out_base = Path(config["input_dir"]) / "American Stories"
    out_base.mkdir(parents=True, exist_ok=True)

    # convert once to pandas for efficient grouping
    logger.info("Converting dataset to pandas dataframe")
    df = ds.to_pandas()
    logger.info("DataFrame converted to pandas dataframe")

    # ensure date is a string suitable for filenames
    df["date"] = df["date"].astype(str)
    logger.info("Date column converted to string")

    for date, g in df.groupby("date"):
        out_path = out_base / f"{date}"
        write_csv(g, out_path, index=True)

def calc_stats():
    res = {}
    for f in Path(config["input_dir"] / "American Stories" / "orig").glob("*.csv"):
        logger.info(f"Calculating stats for {f.stem}")
        df = pd.read_csv(f)
        res[f.stem] = df['newspaper_name'].value_counts()
    df =  pd.DataFrame(res).fillna(0).reset_index()
    
    parsed_cols = df['newspaper_name'].apply(parse_newspaper_name)
    df = parsed_cols.join(df.drop(columns=['newspaper_name']))
    
    # Geocode places to get coordinates
    df = geocode_places(df)
    
    # Create scatter map plot for dates between Sep 1 and Nov 2, 1896
    create_date_range_map(df)
    
    # Create scatter map plot for unique newspapers per location
    create_unique_newspapers_map(df)
    
    return df
    

# Parse newspaper_name to extract name, place, and years
def parse_newspaper_name(name_str):
    logger.info(f"Parsing newspaper name: {name_str}")
    """Parse newspaper name string to extract name, place, and years."""
    if pd.isna(name_str):
        return pd.Series({'newspaper_name_clean': None, 'place': None, 'years': None, 'has_volume': False})
    
    name_str = str(name_str)
    
    # Newspaper name always ends with a dot, so split on first dot
    if '.' not in name_str:
        return pd.Series({'newspaper_name_clean': name_str.strip(), 'place': None, 'years': None, 'has_volume': False})
    
    name_part, rest_part = name_str.split('.', 1)
    clean_name = name_part.strip()
    
    # Extract [volume], place, and years in a single regex
    # Pattern captures all three components: [volume] (anywhere), place in parentheses, years
    pattern = r'(?P<volume>\[volume\])|\((?P<place>[^)]+)\)|(?P<years>\d{4}(?:-\d{4})?)'
    matches = re.finditer(pattern, rest_part, re.IGNORECASE)
    has_volume = False
    place = ''
    years = ''
    for m in matches:
        if m.group('volume'):
            has_volume = True
        if m.group('place'):
            place = m.group('place')
        if m.group('years'):
            years = m.group('years')
        
    return pd.Series({
        'newspaper_name_clean': clean_name,
        'place': place,
        'years': years,
        'has_volume': has_volume
    })


def geocode_place(place_str, api_key):
    """
    Geocode a place string using Google Maps Geocoding API.
    
    Args:
        place_str: Place string to geocode
        api_key: Google Maps API key
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if geocoding fails
    """
    if pd.isna(place_str) or not place_str or place_str.strip() == '':
        return None, None
    
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': place_str.strip(),
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK' and data['results']:
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            logger.warning(f"Geocoding failed for '{place_str}': {data.get('status', 'Unknown error')}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error geocoding '{place_str}': {e}")
        return None, None


def geocode_places(df):
    """
    Add latitude and longitude columns by geocoding the place column.
    
    Args:
        df: DataFrame with a 'place' column
        
    Returns:
        DataFrame with added 'latitude' and 'longitude' columns
    """
    api_key = config.get('google_api_key')

    
    logger.info(f"Geocoding {len(df)} places...")
    
    latitudes = []
    longitudes = []
    
    # Get unique places to avoid duplicate API calls
    unique_places = df['place'].dropna().unique()
    place_coords = {}
    
    for i, place in enumerate(unique_places):
        logger.info(f"Geocoding place: {place} {i}/{len(unique_places)}")
        if place and place.strip():
            lat, lng = geocode_place(place, api_key)
            place_coords[place] = (lat, lng)
            
            # Rate limiting: Google allows 50 requests per second, but we'll be conservative
            if (i + 1) % 10 == 0:
                time.sleep(0.1)  # Small delay every 10 requests
        
    
    # Map coordinates back to dataframe
    for place in df['place']:
        if pd.isna(place) or not place or place.strip() == '':
            latitudes.append(None)
            longitudes.append(None)
        else:
            lat, lng = place_coords.get(place, (None, None))
            latitudes.append(lat)
            longitudes.append(lng)
    
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    
    logger.info("Geocoding completed.")
    return df


def create_date_range_map(df):
    """
    Filter date columns between Sep 1 and Nov 2, 1896, group by coordinates,
    and create a scatter map plot on a US map.
    
    Args:
        df: DataFrame with date columns (format "yyyy-mm-dd") and latitude/longitude columns
    """
    # Define date range
    start_date = datetime(1896, 9, 1)
    end_date = datetime(1896, 11, 2)
    
    # Get all date columns (format "yyyy-mm-dd")
    date_columns = []
    for col in df.columns:
        try:
            date_obj = datetime.strptime(col, "%Y-%m-%d")
            if start_date <= date_obj <= end_date:
                date_columns.append(col)
        except (ValueError, TypeError):
            # Not a date column, skip
            continue
    
    if not date_columns:
        logger.warning("No date columns found in the specified range (1896-09-01 to 1896-11-02)")
        return
    
    logger.info(f"Found {len(date_columns)} date columns in range: {min(date_columns)} to {max(date_columns)}")
    
    # Sum the date columns for each row
    df['date_range_sum'] = df[date_columns].sum(axis=1)
    
    # Filter out rows without coordinates
    df_coords = df[(df['latitude'].notna()) & (df['longitude'].notna())].copy()
    
    if len(df_coords) == 0:
        logger.warning("No rows with valid coordinates found")
        return
    
    # Group by longitude and latitude, summing the counts
    grouped = df_coords.groupby(['longitude', 'latitude'], as_index=False).agg({
        'date_range_sum': 'sum'
    })
    
    logger.info(f"Grouped into {len(grouped)} unique locations")
    
    # Create scatter map plot
    fig = px.scatter_mapbox(
        grouped,
        lat='latitude',
        lon='longitude',
        size='date_range_sum',
        color='date_range_sum',
        color_continuous_scale='Viridis',
        size_max=50,
        zoom=3,
        center={"lat": 39.8283, "lon": -98.5795},  # Center of US
        mapbox_style="open-street-map",
        title="Newspaper Coverage: Sep 1 - Nov 2, 1896",
        labels={'date_range_sum': 'Article Count'},
        hover_data={'latitude': True, 'longitude': True}
    )
    
    fig.update_layout(
        height=800,
        width=1200
    )
    
    return fig


def create_unique_newspapers_map(df):
    """
    Filter date columns between Sep 1 and Nov 2, 1896, count unique newspapers
    per location (only newspapers with articles in that range), and create a scatter map plot.
    
    Args:
        df: DataFrame with date columns (format "yyyy-mm-dd") and latitude/longitude columns
    """
    # Define date range
    start_date = datetime(1896, 9, 1)
    end_date = datetime(1896, 11, 2)
    
    # Get all date columns (format "yyyy-mm-dd")
    date_columns = []
    for col in df.columns:
        try:
            date_obj = datetime.strptime(col, "%Y-%m-%d")
            if start_date <= date_obj <= end_date:
                date_columns.append(col)
        except (ValueError, TypeError):
            # Not a date column, skip
            continue
    
    if not date_columns:
        logger.warning("No date columns found in the specified range (1896-09-01 to 1896-11-02)")
        return None
    
    logger.info(f"Found {len(date_columns)} date columns in range: {min(date_columns)} to {max(date_columns)}")
    
    # Sum the date columns for each row to check if newspaper has articles in range
    df['date_range_sum'] = df[date_columns].sum(axis=1)
    
    # Filter to only newspapers with articles in the date range and valid coordinates
    df_filtered = df[
        (df['date_range_sum'] > 0) & 
        (df['latitude'].notna()) & 
        (df['longitude'].notna())
    ].copy()
    
    if len(df_filtered) == 0:
        logger.warning("No newspapers with articles in date range and valid coordinates found")
        return None
    
    # Identify newspaper column (use 'newspaper_name_clean' if available, otherwise use 'index')
    # After reset_index(), the original index becomes 'index' column
    if 'newspaper_name_clean' in df_filtered.columns:
        newspaper_col = 'newspaper_name_clean'
    elif 'index' in df_filtered.columns:
        newspaper_col = 'index'
    else:
        # Reset index to make it a column if needed
        df_filtered = df_filtered.reset_index()
        if 'index' in df_filtered.columns:
            newspaper_col = 'index'
        else:
            logger.warning("Could not identify newspaper column")
            return None
    
    # Group by longitude and latitude, counting unique newspapers
    grouped = df_filtered.groupby(['longitude', 'latitude'], as_index=False).agg({
        newspaper_col: 'nunique'
    })
    
    # Rename the count column
    grouped = grouped.rename(columns={newspaper_col: 'unique_newspapers'})
    
    logger.info(f"Grouped into {len(grouped)} unique locations with {grouped['unique_newspapers'].sum()} total newspaper-location pairs")
    
    # Create scatter map plot with equal-area projection
    fig = px.scatter_geo(
        grouped,
        lat='latitude',
        lon='longitude',
        size='unique_newspapers',
        color='unique_newspapers',
        color_continuous_scale='Plasma',
        size_max=10,
        scope='usa',  # Focus on USA
        projection='albers usa',  # Equal-area projection for USA
        title="Unique Newspapers per Location: Sep 1 - Nov 2, 1896",
        labels={'unique_newspapers': 'Number of Unique Newspapers'},
        hover_data={'latitude': True, 'longitude': True}
    )
    
    # Reduce colorbar size
    fig.update_traces(
        marker=dict(
            line=dict(width=0),
            colorbar=dict(
                len=0.3,  # Reduce height to 30% of plot
                thickness=15,  # Reduce thickness
                x=1.02  # Position slightly to the right
            )
        )
    )
    
    fix_and_write(fig, "unique_newspapers_map_sep_nov_1896")
    
    return fig


if __name__ == "__main__":
    df = calc_stats()