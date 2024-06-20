import logging
import os
import numpy as np
from pint import UnitRegistry


def cast_to_pixels(unit_str_src, length_float_src, unit_str_tgt, scale_tgt):
    """
    Converts a length from its unit to the viewer's unit and then to pixels.

    Args:
    - unit_str_src: str The unit of the length to convert (ex: the pillar's unit).
    - length_float_src: float The length to convert (ex: the diameter of the pillar).
    - unit_str_tgt: str The unit of the viewer (ex: the calibration unit).
    - scale_tgt: float The scale factor to convert the length to pixels (the calibration of the scalebar).

    Returns:
    - int The length in pixels.
    """
    ureg = UnitRegistry()

    unit_tgt = ureg.parse_expression(unit_str_tgt)
    unit_src = ureg.parse_expression(unit_str_src)

    length_src = length_float_src * unit_src
    length_tgt = length_src.to(unit_tgt)

    return int(length_tgt.magnitude / scale_tgt)


def apply_lut(tags, to_pass, classes):
    """
    tags: Layer that will receive the LUT.
    to_pass: List of gray levels to which we will have to bind a color.
    classes: Dictionary of colors to bind to the gray levels.
    """
    lut = {i: (0.0, 0.0, 0.0, 1.0) for i in range(256)}
    
    for index, level in enumerate(to_pass):
        color = classes[level]
        tpl = [float(color[i])/255.0 for i in range(3)] + [1.0]
        lut[index+1] = tuple(tpl)

    tags.color = lut


def merge_close_points(path, min_distance, dot_threshold=-0.7):
    """
    Merge points in a 2D path that are closer than a specified minimum distance.
    Points are merged into their average position. The operation preserves the path order
    and does not merge non-consecutive points if the path crosses over itself.

    Parameters:
    - path: A list or numpy array of 2D points (e.g., [[x1, y1], [x2, y2], ...]).
    - min_distance: The minimum allowed distance between consecutive points.

    Returns:
    - A new list of points with no consecutive points closer than the minimum distance.
    """
    # Convert path to numpy array for easier distance computations
    merged_path = [path[0]]

    for i in range(1, len(path)):
        current_point = path[i]
        last_merged_point = merged_path[-1]
        distance = np.linalg.norm(last_merged_point - current_point)

        if distance < min_distance:
            potential_merge = (last_merged_point + current_point) / 2
            
            if len(merged_path) >= 2:
                A = merged_path[-2]
                B = last_merged_point
                C = potential_merge
                
                AB = B - A
                AC = C - A
                dot_product = np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))
                
                if dot_product >= dot_threshold:
                    merged_path[-1] = potential_merge
                else:
                    merged_path.append(current_point)
            else:
                merged_path[-1] = potential_merge
        else:
            merged_path.append(current_point)

    return np.array(merged_path)


def smooth_path_2d(points, window_size=3):
    """
    Smooth a path of 2D points using a simple moving average with a sliding window.

    Parameters:
    - points: A NumPy array of 2D points, shape (n_points, 2).
    - window_size: The size of the sliding window for the moving average.

    Returns:
    - A NumPy array of the smoothed 2D points.
    """
    if window_size < 2:
        return points 

    if window_size % 2 == 0:
        window_size += 1

    pad_width = window_size // 2
    padded_points = np.pad(points, ((pad_width, pad_width), (0, 0)), mode='edge')

    smoothed_points = np.zeros_like(points)

    for i in range(len(points)):
        start_index = i
        end_index = i + window_size
        smoothed_points[i] = np.mean(padded_points[start_index:end_index], axis=0)

    return smoothed_points


def setup_logger(file_path):
    base_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    
    logger = logging.getLogger(file_name_without_extension)
    logger.setLevel(logging.INFO) 
    
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger



if __name__ == "__main__":
    path = '/home/benedetti/Desktop/path.npy'
    vertices_raw = np.load(path)
    print(vertices_raw)
    print(len(vertices_raw))

    vertices_simplified = merge_close_points(vertices_raw, 2.0)
    print(vertices_simplified)
    print(len(vertices_simplified))