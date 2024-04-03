from typing import Tuple

import numpy as np
from dagster import asset, op, MaterializeResult, MetadataValue, FilesystemIOManager, Definitions, \
    AssetExecutionContext, Config
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from pandas import DataFrame
import json
from os import path, listdir
from PIL import Image, ImageDraw
import base64
import colorsys
import random
from io import BytesIO
from shapely.geometry import mapping, Polygon
import cv2
import pyvista as pv

from ...entities.bbox import BBox
from ...entities.bbox_grouping import BBoxGrouping
from ...entities.orientation import Orientation
from ...helpers.Equirec2Perspec import Equirectangular
from ...CONSTANTS import ROOT_PATH, CLASS_ID_TO_CLASS_NAME

Point3D = Tuple[float, float, float]


class StreetLevelConfig(Config):
    results_path: str = 'results/results.json'
    mesh_path: str = 'lidar/Mesh/MID_resolution.obj'
    polygon_spacing: int = 10


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
def detected_objects_street_level(context: AssetExecutionContext, config: StreetLevelConfig) -> DataFrame:
    with open(path.join(ROOT_PATH, config.results_path), 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data, 'objects', 'file_name', max_level=0)
    df['class_name'] = df['class'].apply(lambda x: CLASS_ID_TO_CLASS_NAME[x])
    context.add_output_metadata({
        'schema': MetadataValue.md(df.dtypes.to_markdown()),
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
    context.add_output_metadata({
        'schema': MetadataValue.md(df_grouped.dtypes.to_markdown())
    })
    return df_grouped


@asset
def reference_file(context: AssetExecutionContext) -> DataFrame:
    PHOTO_PATH = path.join(ROOT_PATH, 'photo')
    ref = pd.read_csv(path.join(PHOTO_PATH, 'reference.csv'), sep='\t')
    return ref


def process_group(
        file_group: DataFrame,
        orientation: Orientation,
        origin: Tuple[float, float, float],
        image_width: int,
        image_height: int,
        spacing: int) -> DataFrame:
    best_rows_df = pd.DataFrame()
    for group_idx, group in file_group.groupby('group_idx'):
        group = group[group['polygon'] != '']
        if group.empty:
            continue
        best_row = group.loc[group['score'].idxmax()]
        polygon = Polygon(best_row['polygon']['coordinates'][0])
        polygon_3d = [
            point_to_3d_line(
                (int(point[0]), int(point[1])),
                orientation,
                origin,
                image_width,
                image_height)
            for point in list(zip(*polygon.exterior.coords.xy))[0::spacing]]
        best_row['polygon_3d'] = polygon_3d
        best_row = best_row.drop('group_idx')
        best_row['origin'] = origin
        best_rows_df = best_rows_df.append(best_row)
    return best_rows_df


@asset
def best_lines_3d(
        context: AssetExecutionContext,
        config: StreetLevelConfig,
        grouped_detected_objects: DataFrame,
        reference_file: DataFrame) -> DataFrame:
    PHOTO_PATH = path.join(ROOT_PATH, 'photo')
    args_list = []
    for file_name, file_group in grouped_detected_objects.groupby('file_name'):
        image_width, image_height = get_image_dimensions(path.join(PHOTO_PATH, str(file_name)))
        row = reference_file.loc[reference_file['file_name'] == str(file_name).split('.')[0]]
        orientation = Orientation(
            roll=-float(row['roll[deg]']),
            pitch=float(row['pitch[deg]']),
            heading=float(row['heading[deg]'] + 90)).rads
        origin = (float(row['projectedX[m]']), float(row['projectedY[m]']), float(row['projectedZ[m]']))
        args_list.append((file_group, orientation, origin, image_width, image_height, config.polygon_spacing))
    with parallel_backend('loky', n_jobs=-1):
        results = Parallel()(delayed(process_group)(*args) for args in args_list)
    best_rows_df = pd.concat(results)
    best_rows_df.to_csv(path.join(ROOT_PATH, 'results', 'best_rows.csv'))
    context.add_output_metadata({
        'schema': MetadataValue.md(best_rows_df.dtypes.to_markdown())
    })
    context.log.info(best_rows_df.head(4))
    return best_rows_df


@asset
def point_and_mesh_intersection(
        context: AssetExecutionContext,
        config: StreetLevelConfig,
        best_lines_3d: DataFrame) -> DataFrame:
    mesh = pv.read(path.join(ROOT_PATH, config.mesh_path))
    best_lines_3d = best_lines_3d[best_lines_3d['polygon_3d'].apply(lambda x: len(x) > 1)]
    all_objects = best_lines_3d[['origin', 'polygon_3d']].values
    all_lines = []
    for origin, polygon_3d in all_objects:
        for point in polygon_3d:
            all_lines.append((np.array(origin), np.array(point)))
    context.log.info(f'Number of lines: {len(all_lines)}')

    points, indices, cells = mesh.multi_ray_trace(
        [np.array(x[0]) for x in all_lines], [np.array(x[1]) for x in all_lines],
        first_point=True)

    context.log.info(f'Number of points: {len(points)}')
    # collect points back to polygon_3d
    for i, (origin, polygon_3d) in enumerate(all_objects):
        for j, point in enumerate(polygon_3d):
            if points[i * len(polygon_3d) + j] is not None:
                polygon_3d[j] = points[i * len(polygon_3d) + j]
        best_lines_3d.loc[i, 'polygon_3d'] = polygon_3d

    best_lines_3d.to_csv(path.join(ROOT_PATH, 'results', 'intersecting_lines.csv'))

    context.add_output_metadata({
        'schema': MetadataValue.md(best_lines_3d.dtypes.to_markdown())
    })

    return best_lines_3d


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


@op
def point_to_3d_line(
        point: Tuple[int, int],
        orientation_rads: Orientation,
        origin: Point3D,
        image_width: int,
        image_height: int,
        length=1) -> Point3D:
    x, y = point

    theta = 2 * np.pi * (image_width - x) / image_width  # azimuthal angle
    phi = np.pi * y / image_height

    x_cartesian = np.sin(phi) * np.cos(theta)
    y_cartesian = np.sin(phi) * np.sin(theta)
    z_cartesian = np.cos(phi)

    # Roll
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(orientation_rads.roll), -np.sin(orientation_rads.roll)],
        [0, np.sin(orientation_rads.roll), np.cos(orientation_rads.roll)]])

    # Pitch
    rotation_matrix_y = np.array([
        [np.cos(orientation_rads.pitch), 0, np.sin(orientation_rads.pitch)],
        [0, 1, 0],
        [-np.sin(orientation_rads.pitch), 0, np.cos(orientation_rads.pitch)]])

    # Heading (Yaw)
    rotation_matrix_z = np.array([
        [np.cos(orientation_rads.heading), np.sin(orientation_rads.heading), 0],
        [-np.sin(orientation_rads.heading), np.cos(orientation_rads.heading), 0],
        [0, 0, 1]])

    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    result: np.ndarray = np.dot(rotation_matrix, np.array([x_cartesian, y_cartesian, z_cartesian]))
    if length != 1:
        result = result * length

    result = result + np.array(origin)
    return result[0], result[1], result[2]


@op
def get_image_dimensions(image_path):
    with open(image_path, 'rb') as img_file:
        img_file.seek(163)
        a = img_file.read(2)
        height = (a[0] << 8) + a[1]
        a = img_file.read(2)
        width = (a[0] << 8) + a[1]
    return width, height


@asset
def visualize(grouped_detected_objects: DataFrame) -> MaterializeResult:
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
