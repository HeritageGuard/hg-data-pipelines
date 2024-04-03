from dagster import asset, op, AssetExecutionContext, MetadataValue, Definitions
import pandas as pd
import json
from os import path
import uuid
from shapely.geometry import Polygon

from ...CONSTANTS import ROOT_PATH, RESULTS_FILE, CLASS_ID_TO_CLASS_NAME

@op
def parse_polygon(polygon) -> str:
    try:
        return Polygon(polygon['coordinates'][0]).to_wkt()
    except:
        return Polygon().to_wkt()

@asset(io_manager_key='duck_io_manager')
def detected_objects(context: AssetExecutionContext) -> pd.DataFrame:
    with open(path.join(ROOT_PATH, RESULTS_FILE), 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data, 'objects', 'file_name', max_level=0)
    df['class_name'] = df['class'].apply(lambda x: CLASS_ID_TO_CLASS_NAME[x])
    df['id'] = df['file_name'].map(lambda _: uuid.uuid4())
    df['polygon'] = df['polygon'].map(parse_polygon)
    context.add_output_metadata({
        'detected_classes': MetadataValue.md(df['class_name'].value_counts().to_markdown())
    })
    return df


