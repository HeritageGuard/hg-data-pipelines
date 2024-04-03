import os
import cv2
from dagster import op, asset, Config, Output
from ...helpers.Equirec2Perspec import Equirectangular
from ...CONSTANTS import ROOT_PATH


class ProjectionsConfig(Config):
    fov: int = 60
    phi: int = -10
    width: int = 1000
    height: int = 1000
    subdivisions: int = 20


def get_angle_subdivisions(x):
    # Calculate the angle increment for each subdivision
    angle_increment = 360 / x
    # Generate the list of subdivisions using list comprehension
    subdivisions = [int(i * angle_increment) for i in range(x)]
    return subdivisions


@op
def tls_photos():
    image_path = os.path.join(ROOT_PATH, 'data/tls_images')
    image_names= []
    for (dir_path, _, file_names) in os.walk(image_path):
        for file_name in file_names:
            image_names.append([dir_path, file_name])
    return image_names


@op
def generate_image(config: ProjectionsConfig, equ, file_name, theta):
    img = equ.GetPerspective(config.fov, theta, config.phi, config.height, config.width)
    image_path = os.path.join(ROOT_PATH, 'data/projections', str(theta) + '_' + file_name)
    cv2.imwrite(image_path, img)
    return Output(file_name)


@asset
def generate_projections(config: ProjectionsConfig):
    thetas = get_angle_subdivisions(config.subdivisions)

    for dir_path, file_name in tls_photos():
        equ = Equirectangular(dir_path + '/' + file_name)
        for theta in thetas:
            generate_image(config, equ, file_name, theta)

