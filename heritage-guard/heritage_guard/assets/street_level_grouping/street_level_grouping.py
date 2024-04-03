from dagster import asset, op, MaterializeResult, MetadataValue, FilesystemIOManager, Definitions, AssetExecutionContext
import pandas as pd
from pandas import DataFrame
import json
from os import path, listdir
from PIL import Image, ImageDraw
import base64
import colorsys
import random
from io import BytesIO
from shapely.geometry import mapping
import cv2

from ...helpers.bbox import BBox
from ...helpers.bbox_grouping import BBoxGrouping
from ...helpers.Equirec2Perspec import Equirectangular
from ...CONSTANTS import ROOT_PATH, RESULTS_FILE, CLASS_ID_TO_CLASS_NAME


@asset
def results_file(context: AssetExecutionContext):
    fov = 60
    width = 1000
    height = 1000
    phi = -10
    results = []
    photos_dir = path.join(ROOT_PATH, 'data', 'tls_images')
    predictions_dir = path.join(ROOT_PATH, 'data', 'predictions')
    results_dir = path.join(ROOT_PATH, 'results')
    for photo_file_name in listdir(photos_dir):
        context.log.info(f'Processing {photo_file_name}')
        bboxes = []
        polygons = []
        scores = []
        classes = []
        equ = Equirectangular(path.join(photos_dir, photo_file_name))
        for filename in listdir(predictions_dir):
            # Check if the file is a JSON file
            if filename.endswith(photo_file_name[:-4] + ".json"):
                filepath = path.join(predictions_dir, filename)
                theta = int(filename.split('_')[0])
                # Read and decode the JSON file
                with open(filepath, 'r') as json_file:
                    decoded_object = json.load(json_file)
                    bboxes += [equ.GetBboxInverse(fov, theta, phi, height, width, bbox) for bbox in
                               decoded_object['bboxes']]
                    classes += decoded_object['labels']
                    scores += decoded_object['scores']
                    [polygons.extend(result) if isinstance(result, tuple) else polygons.append(result) for result in
                     [equ.GetPolygonInverse(fov, theta, phi, height, width, mask) for mask in
                      decoded_object['masks']]]
        objects = []
        for index, bbox in enumerate(bboxes):
            objects.append({
                'bbox': bbox,
                'polygon': mapping(polygons[index]) if polygons[index] is not None else '',
                'score': scores[index],
                'class': classes[index]
            })
        results.append({
            'file_name': photo_file_name,
            'objects': objects
        })

        img = equ.draw_bbox_360(bboxes, polygons, scores)
        cv2.imwrite(path.join(results_dir, 'photo', photo_file_name), img)
    with open(path.join(results_dir, 'results.json'), "w") as outfile:
        outfile.write(json.dumps(results))

@asset
def detected_objects_street_level(context: AssetExecutionContext) -> DataFrame:
    with open(path.join(ROOT_PATH, 'results', RESULTS_FILE), 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data, 'objects', 'file_name', max_level=0)
    df['class_name'] = df['class'].apply(lambda x: CLASS_ID_TO_CLASS_NAME[x])
    context.add_output_metadata({
        'detected_classes': MetadataValue.md(df['class_name'].value_counts().to_markdown())
    })
    return df


@asset
def grouped_detected_objects(context: AssetExecutionContext, detected_objects_street_level: DataFrame) -> DataFrame:
    df = detected_objects_street_level
    grouped_df = []

    for _, group in df.groupby('file_name'):
        group = group.reset_index()
        instances = []
        bboxes = group['bbox'].values

        for index, bbox in enumerate(bboxes):
            instance = BBox(bbox, group.loc[index, 'score'], group.loc[index, 'class'])
            instances.append(instance)

        bbox_grouping = BBoxGrouping(instances, 0.01)
        bbox_grouping.calculate_similarity_matrix()
        bbox_grouping.group_bboxes()

        # New code starts here
        bbox_to_group = {}
        for group_index, group_id in enumerate(bbox_grouping.groups):
            for bbox_index in group_id:
                bbox_to_group[bbox_index] = group_index

        # Map each bbox to its group index
        group['group_idx'] = group.index.to_series().map(bbox_to_group)
        # Append the modified group to the grouped_df list
        grouped_df.append(group)

    # Combine all the grouped data into a single DataFrame
    df_grouped = pd.concat(grouped_df)

    df_grouped.to_csv('../../grouped_objects.csv')
    return df_grouped


@op
def generate_unique_colors(n):
    """
    Generates n unique colors.

    :param n: The number of unique colors to generate.
    :return: A list of unique colors in RGB format.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Vary the hue to get a unique color
        saturation = 0.7  # Keep saturation and value constant
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb = tuple([int(x * 255) for x in rgb])  # Convert to RGB format used by PIL
        colors.append(rgb)
    random.shuffle(colors)
    return colors

@asset
def visualize(context: AssetExecutionContext, grouped_detected_objects: DataFrame) -> MaterializeResult:
    image_data = None
    for file_index, file_name in enumerate(grouped_detected_objects['file_name'].unique()):
        first_file_data = grouped_detected_objects.loc[grouped_detected_objects['file_name'] == file_name]
        image = Image.open(path.join(ROOT_PATH, 'data', 'tls_images', file_name))
        image = image.convert('RGB')
        draw = ImageDraw.Draw(image)
        colors = generate_unique_colors(len(first_file_data))

        for group_idx, group_data in first_file_data[first_file_data['score'] > 0.9].groupby('group_idx'):
            for bbox in group_data['bbox']:
                draw.rectangle(bbox, outline=colors[group_idx], width=5)
        width, height = image.size
        image = image.resize((int(width / 5), int(height / 5)))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image.save(path.join(ROOT_PATH, 'results', 'photo', 'merged', file_name))
        if file_index == 0:
            image_data = base64.b64encode(buffered.getvalue()).decode()

    return MaterializeResult(
        metadata={
            'image_preview': MetadataValue.md(f"![img](data:image/png;base64,{image_data})")
        })


defs = Definitions(
    assets=[detected_objects_street_level, grouped_detected_objects, visualize],
    resources={
        "io_manager": FilesystemIOManager(based_dir='/home/domas/projects/HG Domo')
    }
)
