import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Orientation:
    """ Orientation of the sphere """
    roll: float
    pitch: float
    heading: float
    rad: bool = field(default=False)

    @property
    def rads(self):
        return Orientation(
            roll=np.deg2rad(self.roll),
            pitch=np.deg2rad(self.pitch),
            heading=np.deg2rad(self.heading),
            rad=True
        )

    def to_tuple(self, heading_offset: float = None) -> Tuple[float, float, float]:
        return self.heading if heading_offset is None else self.heading + heading_offset, self.pitch, self.roll
