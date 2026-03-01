from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float
    intensity: float = 0.0
    classification: int = 0
    r: int = 0
    g: int = 0
    b: int = 0
    return_num: int = 0