import os
import sys
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from pycocotools import mask
import numpy as np


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


def compute_calib_matrix(width, height, FOV):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], np.float32)

    return K, np.linalg.inv(K)


def compute_rotation_matrix(THETA, PHI):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)

    R1, _ = cv2.Rodrigues(y_axis * np.radians(-THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(-PHI))

    return R2 @ R1


def transform_coords(coords, K_inv, R):
    coords_3d = coords @ K_inv.T
    return coords_3d @ R.T

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        K, K_inv = compute_calib_matrix(width, height, FOV)
        R = compute_rotation_matrix(THETA, PHI)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

        xyz_transformed = transform_coords(xyz, K_inv, R)

        lonlat = xyz2lonlat(xyz_transformed)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)

        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

    def GetBboxInverse(self, FOV, THETA, PHI, height, width, bbox):
        K, K_inv = compute_calib_matrix(width, height, FOV)
        R = compute_rotation_matrix(THETA, PHI)

        bbox_3d = np.array([
            [bbox[0], bbox[1], 1],
            [bbox[2], bbox[1], 1],
            [bbox[0], bbox[3], 1],
            [bbox[2], bbox[3], 1]
        ])

        bbox_3d_transformed = transform_coords(bbox_3d, K_inv, R)

        lonlat = xyz2lonlat(bbox_3d_transformed)
        bbox_360 = lonlat2XY(lonlat, shape=self._img.shape).astype(np.int32)

        x1, y1 = np.min(bbox_360[:, 0]), np.min(bbox_360[:, 1])
        x2, y2 = np.max(bbox_360[:, 0]), np.max(bbox_360[:, 1])

        image_height, image_width, _ = self._img.shape
        #
        # if abs(x2 - x1) > image_width // 2:
        #     # Bbox crosses the seam, so we split it
        #     bbox1 = [0, y1, np.max(bbox_360[bbox_360[:, 0] <= image_width // 2, 0]), y2]
        #     # bbox2 = [np.min(bbox_360[bbox_360[:, 0] > image_width // 2, 0]), y1, image_width - 1, y2]
        #     return bbox1
        return [int(x1), int(y1), int(x2), int(y2)]

    def GetPolygonInverse(self, FOV, THETA, PHI, height, width, polygon_coco):
        binary_mask = mask.decode(polygon_coco)

        # Convert binary mask to polygon coordinates
        contours, hierarchy = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0 or len(contours[0][:, 0, :]) < 3:
            return None
        polygon_flat = Polygon(contours[0][:, 0, :])

        K, K_inv = compute_calib_matrix(width, height, FOV)
        R = compute_rotation_matrix(THETA, PHI)

        x, y = polygon_flat.exterior.coords.xy
        z = np.ones_like(x)
        xyz = np.stack((x, y, z), axis=-1)
        xyz = transform_coords(xyz, K_inv, R)

        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.int32)
        return Polygon(XY)
        # return self.handle_split_polygons(XY)

    def handle_split_polygons(self, XY):
        height, width, _ = self._img.shape

        split_polygons = []

        for i in range(len(XY) - 1):
            if abs(XY[i, 0] - XY[i + 1, 0]) > width // 2:
                mid_y = (XY[i, 1] + XY[i + 1, 1]) // 2
                if XY[i, 0] < width // 2:
                    split_polygons.append([(width - 1, mid_y), (0, mid_y)])
                else:
                    split_polygons.append([(0, mid_y), (width - 1, mid_y)])

        if split_polygons:
            # Split the polygon at the seam
            left_side = [pt for pt in XY if pt[0] <= width // 2]
            right_side = [pt for pt in XY if pt[0] > width // 2]

            # If a polygon part is on the boundary, append the splitting coordinates to it
            # if left_side:
            #     left_side.extend(split_polygons[0])
            # if right_side:
            #     right_side.extend(split_polygons[1])

            # Concatenate to form separate polygons
            XY1 = np.array(left_side, dtype=np.int32)
            XY2 = np.array(right_side, dtype=np.int32)

            return Polygon(XY1), Polygon(XY2)

        else:
            return Polygon(XY)
    def draw_bbox_360(self, bboxes, polygons, scores):
        img_with_bbox = self._img.copy()
        for main_index in range(len(bboxes)):
            bbox = bboxes[main_index]
            # cv2.rectangle(img_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, int(255*scores[main_index]), 0), 2)
            polygon = polygons[main_index]
            if polygon is None:
                continue
            xy_arrays = polygon.exterior.xy
            xy_polygon = []
            if len(xy_arrays[0]) < 3:
                print('Polygon too small', xy_arrays)
            for index in range(len(xy_arrays[0])):
                xy_polygon.append([xy_arrays[0][index], xy_arrays[1][index]])
            xy_polygon = np.array(xy_polygon, np.int32)
            xy_polygon = xy_polygon.reshape((-1, 1, 2))
            cv2.polylines(img_with_bbox, [xy_polygon], True, (0, int(255*scores[main_index]), int(255-(255*scores[main_index]))), 4)
        return img_with_bbox

    def draw_bbox_flat(self, persp_img, bbox):
        img_with_bbox = persp_img.copy()
        cv2.rectangle(img_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow('flat_with_bbox', img_with_bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

