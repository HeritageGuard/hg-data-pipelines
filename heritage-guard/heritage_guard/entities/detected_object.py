from dataclasses import dataclass
from shapely.geometry import Polygon

from .bbox import BBox
from object_class import ObjectClass


@dataclass
class DetectedObject:
    bbox: BBox
    polygon: Polygon
    score: float
    object_class: ObjectClass
