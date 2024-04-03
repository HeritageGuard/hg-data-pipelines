from dagster import load_assets_from_package_module
from . import tls_photo_processing, street_level_grouping, detected_objects_processing

# Dagster asset loading from modules
street_level_grouping_assets = load_assets_from_package_module(
    package_module=street_level_grouping,
    group_name="street_level_grouping_assets"
)

tls_photo_processing_assets = load_assets_from_package_module(
    package_module=tls_photo_processing,
    group_name="tls_photo_processing_assets"
)

detected_objects_processing_assets = load_assets_from_package_module(
    package_module=detected_objects_processing,
    group_name="detected_objects_processing_assets"
)