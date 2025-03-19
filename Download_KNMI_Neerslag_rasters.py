import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from requests import Session
from configparser import ConfigParser
from datetime import datetime, timedelta
import xarray as xr
import rioxarray
import rasterio
import geopandas as gpd
from sqlalchemy import create_engine, Integer, Column, Date, Numeric, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from geoalchemy2 import Geometry

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

try:
    import geoalchemy2
except ImportError:
    logger.error("The geoalchemy2 package is required. Install it using 'pip install geoalchemy2'")
    raise

def connect_to_database(config: ConfigParser) -> Any:
    db_config = config['database']
    connection_string = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    try:
        engine = create_engine(connection_string)
        connection = engine.connect()
        logger.info("Successfully connected to the database")
        return connection, engine
    except SQLAlchemyError as e:
        logger.error(f"Database connection failed: {e}")
        return None, None

def create_or_replace_table(connection, schema: str, table_name: str, table_definition: str):
    try:
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} ({table_definition});
            TRUNCATE TABLE {schema}.{table_name};
        """))
        logger.info(f"Created or replaced table {schema}.{table_name}")
    except OperationalError as e:
        logger.error(f"Error creating or replacing table {schema}.{table_name}: {e}")
        raise

def download_dataset_file(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
    directory: str,
    overwrite: bool,
) -> Tuple[bool, str]:
    file_path = Path(directory, filename).resolve()
    if not overwrite and file_path.exists():
        logger.info(f"Dataset file '{filename}' was already downloaded.")
        return True, filename

    endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    get_file_response = session.get(endpoint)

    if get_file_response.status_code != 200:
        logger.warning(f"Unable to get file: {filename}")
        logger.warning(get_file_response.content)
        return False, filename

    download_url = get_file_response.json().get("temporaryDownloadUrl")
    return download_file_from_temporary_download_url(download_url, directory, filename)


def download_file_from_temporary_download_url(download_url, directory, filename):
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(f"{directory}/{filename}", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.exception("Unable to download file using download URL")
        return False, filename

    logger.info(f"Downloaded dataset file '{filename}'")
    return True, filename


def list_dataset_files(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    params: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    logger.info(f"Retrieve dataset files with query params: {params}")

    list_files_endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"
    list_files_response = session.get(list_files_endpoint, params=params)

    if list_files_response.status_code != 200:
        raise Exception("Unable to list initial dataset files")

    try:
        list_files_response_json = list_files_response.json()
        dataset_files = list_files_response_json.get("files")
        dataset_filenames = list(map(lambda x: x.get("filename"), dataset_files))
        return dataset_filenames, list_files_response_json
    except Exception as e:
        logger.exception(e)
        raise Exception(e)


def get_max_worker_count(filesizes):
    size_for_threading = 10_000_000  # 10 MB
    average = sum(filesizes) / len(filesizes)
    if average > size_for_threading:
        threads = 1
    else:
        threads = 10
    return threads

def extract_date_from_filename(filename: str) -> datetime:
    parts = filename.split('_')
    for part in parts:
        try:
            return datetime.strptime(part[:8], "%Y%m%d")
        except ValueError:
            continue
    raise ValueError(f"No valid date found in filename: {filename}")

def filter_files_by_date(filenames: List[str], start_date: str, end_date: str) -> List[str]:
    filtered_filenames = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    for filename in filenames:
        try:
            file_dt = extract_date_from_filename(filename)
            if start_dt <= file_dt <= end_dt:
                filtered_filenames.append(filename)
        except ValueError:
            logger.warning(f"Could not extract date from filename: {filename}")

    return filtered_filenames

def convert_netcdf_to_tiff(netcdf_file: str, output_directory: str):
    try:
        dataset = xr.open_dataset(netcdf_file, engine='netcdf4')
        data_var_name = 'prediction'  # Replace with the actual data variable name if different
        data_var = dataset[data_var_name]

        data_var = data_var.rio.write_crs("EPSG:28992")

        tiff_filename = Path(netcdf_file).with_suffix('.tiff').name
        output_path = Path(output_directory) / tiff_filename

        if output_path.exists():
            logger.info(f"TIFF file '{output_path}' already exists. Skipping conversion.")
            dataset.close()
            return

        data_var.rio.to_raster(output_path)
        logger.info(f"Converted '{netcdf_file}' to '{output_path}'")

    except Exception as e:
        logger.exception(f"Failed to convert {netcdf_file} to TIFF: {e}")

    finally:
        if dataset is not None:
            dataset.close()

def get_most_recent_date_from_files(directory: str) -> str:
    most_recent_date = None
    for filename in os.listdir(directory):
        if filename.endswith('.nc'):
            try:
                file_date = extract_date_from_filename(filename)
                if most_recent_date is None or file_date > most_recent_date:
                    most_recent_date = file_date
            except ValueError:
                continue
    if most_recent_date is None:
        return None
    return most_recent_date.strftime('%Y-%m-%d')

def perform_analysis(output_directory: str, gdf: gpd.GeoDataFrame, geometry_column: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    results = []
    avg10_results = []
    avg30_results = []

    for tiff_file in Path(output_directory).glob("*.tiff"):
        date_str = extract_date_from_filename(tiff_file.stem).strftime('%Y-%m-%d')
        tiff_date = datetime.strptime(date_str, '%Y-%m-%d')
        with rasterio.open(tiff_file) as src:
            for _, row in gdf.iterrows():
                geom = row[geometry_column]  # Use the correct geometry column
                location_id = row['location_id']
                earliest_measurement = row['earliest_measurement']

                if geom is None or geom.is_empty:
                    logger.warning(f"Found invalid geometry. Skipping.")
                    continue

                if tiff_date < earliest_measurement - timedelta(days=30):
                    continue

                coords = [(geom.x, geom.y)]
                sampled_value = [val[0] for val in src.sample(coords)][0]  # Convert numpy array to float
                results.append({
                    'location_id': location_id,
                    'geom': geom,
                    'date': tiff_date.date(),  # Convert to date
                    'neerslag_waarde': float(sampled_value)  # Convert to float
                })

    if not results:
        logger.warning("No valid samples were found in the TIFF files.")
        return None, None, None

    sampled_gdf = gpd.GeoDataFrame(results, geometry='geom', crs=gdf.crs)

    # Calculate average values for the last 10 and 30 days
    for location_id in gdf['location_id'].unique():
        location_data = sampled_gdf[sampled_gdf['location_id'] == location_id].sort_values(by='date')
        for i in range(len(location_data)):
            if i >= 9:
                avg10_value = location_data.iloc[i-9:i+1]['neerslag_waarde'].mean()
                avg10_results.append({
                    'location_id': location_id,
                    'geom': location_data.iloc[i]['geom'],
                    'date': location_data.iloc[i]['date'],
                    'avg10_neerslag_waarde': avg10_value
                })
            if i >= 29:
                avg30_value = location_data.iloc[i-29:i+1]['neerslag_waarde'].mean()
                avg30_results.append({
                    'location_id': location_id,
                    'geom': location_data.iloc[i]['geom'],
                    'date': location_data.iloc[i]['date'],
                    'avg30_neerslag_waarde': avg30_value
                })

    avg10_gdf = gpd.GeoDataFrame(avg10_results, geometry='geom', crs=gdf.crs) if avg10_results else None
    avg30_gdf = gpd.GeoDataFrame(avg30_results, geometry='geom', crs=gdf.crs) if avg30_results else None

    return sampled_gdf, avg10_gdf, avg30_gdf

def connect_and_create_tables(engine, schema: str, new_layer: str):
    with engine.begin() as connection:
        # Define table schemas
        main_table_definition = """
        location_id INTEGER,
        date DATE,
        neerslag_waarde NUMERIC,
        geom Geometry(POINT, 28992)
        """
        avg10_table_definition = """
        location_id INTEGER,
        date DATE,
        avg10_neerslag_waarde NUMERIC,
        geom Geometry(POINT, 28992)
        """
        avg30_table_definition = """
        location_id INTEGER,
        date DATE,
        avg30_neerslag_waarde NUMERIC,
        geom Geometry(POINT, 28992)
        """

        # Create or replace tables
        create_or_replace_table(connection, schema, new_layer, main_table_definition)
        create_or_replace_table(connection, schema, f"{new_layer}_10dag", avg10_table_definition)
        create_or_replace_table(connection, schema, f"{new_layer}_30dag", avg30_table_definition)

def insert_data_into_tables(engine, schema: str, new_layer: str, sampled_gdf: gpd.GeoDataFrame, avg10_gdf: gpd.GeoDataFrame, avg30_gdf: gpd.GeoDataFrame):
    with engine.begin() as connection:
        if sampled_gdf is not None:
            logger.info("Inserting data into main table")
            sampled_gdf.to_postgis(name=new_layer, schema=schema, con=engine, if_exists='append', index_label='idx_' + new_layer[:50])
            logger.info(f"Successfully created layer '{schema}.{new_layer}' in the database")

        if avg10_gdf is not None:
            logger.info("Inserting data into 10-day average table")
            avg10_gdf.to_postgis(name=f"{new_layer}_10dag", schema=schema, con=engine, if_exists='append', index_label='idx_' + new_layer[:47] + '_10dag')
            logger.info(f"Successfully created layer '{schema}.{new_layer}_10dag' in the database")

        if avg30_gdf is not None:
            logger.info("Inserting data into 30-day average table")
            avg30_gdf.to_postgis(name=f"{new_layer}_30dag", schema=schema, con=engine, if_exists='append', index_label='idx_' + new_layer[:47] + '_30dag')
            logger.info(f"Successfully created layer '{schema}.{new_layer}_30dag' in the database")

async def main():
    config = ConfigParser()
    config.read('config.ini')
    api_key = config.get('api', 'key')
    download_directory = config.get('paths', 'download_directory')
    output_directory = config.get('paths', 'output_directory')
    start_date = config.get('dates', 'start_date')
    end_date = datetime.now().strftime('%Y-%m-%d')
    schema = config.get('database', 'schema')
    source_layer = config.get('database', 'source_layer')
    new_layer = config.get('database', 'new_layer')
    geometry_column = config.get('database', 'geometry_column')

    dataset_name = "Rd1"
    dataset_version = "5"
    base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
    overwrite = False

    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    if not Path(download_directory).is_dir() or not Path(download_directory).exists():
        Path(download_directory).mkdir(parents=True, exist_ok=True)

    if not Path(output_directory).is_dir() or not Path(output_directory).exists():
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    most_recent_date = get_most_recent_date_from_files(download_directory)
    logger.info(f"Most recent date from files: {most_recent_date}")

    if most_recent_date is None:
        logger.info("No recent files found, using start date from config")
        most_recent_date = start_date
    elif datetime.strptime(most_recent_date, '%Y-%m-%d') > datetime.strptime(start_date, '%Y-%m-%d'):
        start_date = most_recent_date

    logger.info(f"Using start date: {start_date}")

    filenames = []
    max_keys = 500
    next_page_token = None
    file_sizes = []

    params = {
        "maxKeys": str(max_keys),
        "startDate": start_date,
        "endDate": end_date,
    }

    while True:
        dataset_filenames, response_json = list_dataset_files(
            session,
            base_url,
            dataset_name,
            dataset_version,
            params,
        )
        file_sizes.extend(file["size"] for file in response_json.get("files"))
        filenames += dataset_filenames

        next_page_token = response_json.get("nextPageToken")
        if not next_page_token:
            logger.info("Retrieved names of all dataset files")
            break

        params["nextPageToken"] = next_page_token

    filenames = filter_files_by_date(filenames, start_date, end_date)
    logger.info(f"Number of files to download: {len(filenames)}")

    worker_count = get_max_worker_count(file_sizes)
    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = []

    for dataset_filename in filenames:
        future = loop.run_in_executor(
            executor,
            download_dataset_file,
            session,
            base_url,
            dataset_name,
            dataset_version,
            dataset_filename,
            download_directory,
            overwrite,
        )
        futures.append(future)

    future_results = await asyncio.gather(*futures)
    logger.info(f"Finished '{dataset_name}' dataset download")

    failed_downloads = list(filter(lambda x: not x[0], future_results))

    if len(failed_downloads) > 0:
        logger.warning("Failed to download the following dataset files:")
        logger.warning(list(map(lambda x: x[1], failed_downloads)))

    netcdf_files = [os.path.join(download_directory, filename) for filename in filenames]
    for netcdf_file in netcdf_files:
        convert_netcdf_to_tiff(netcdf_file, output_directory)

    connection, engine = connect_to_database(config)
    if connection:
        # Get the earliest measurement timestamp per location_id
        query = f"""
        SELECT location_id, {geometry_column}, MIN(measurement_timestamp) as earliest_measurement
        FROM {schema}.{source_layer}
        GROUP BY location_id, {geometry_column}
        """
        gdf = gpd.read_postgis(query, connection, geom_col=geometry_column)
        if gdf.empty:
            logger.error(f"No data found in the source layer {schema}.{source_layer}")
            connection.close()
            return

        # Ensure earliest_measurement is naive datetime (without timezone)
        gdf['earliest_measurement'] = gdf['earliest_measurement'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)

        connection.close()

        # Perform the analysis before connecting to the database
        sampled_gdf, avg10_gdf, avg30_gdf = perform_analysis(output_directory, gdf, geometry_column)

        if sampled_gdf is None:
            logger.error("No valid samples were generated.")
            return

        # Reconnect to the database and create/replace the tables
        connection, engine = connect_to_database(config)
        if connection:
            connect_and_create_tables(engine, schema, new_layer)

            # Insert the data into the tables
            insert_data_into_tables(engine, schema, new_layer, sampled_gdf, avg10_gdf, avg30_gdf)

            connection.close()

if __name__ == "__main__":
    asyncio.run(main())
