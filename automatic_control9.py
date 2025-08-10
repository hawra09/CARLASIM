#!/usr/bin/env python

# Based on: CARLA automatic vehicle control example
# Original author: German Ros (Intel Labs)
# Modified by: Hawra
# License: MIT (https://opensource.org/licenses/MIT)

"""Improved AV script with pedestrian detection, sensor fusion, and data collection"""

from __future__ import print_function

import argparse
import collections
import copy
import csv
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import threading
import time
import weakref
import queue

import numpy as np
import pygame
import cv2
from PIL import Image
from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q
from ultralytics import YOLO

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from carla import WalkerAIController

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

from sklearn.cluster import DBSCAN

class BoundingBox(object):
    def __init__(self, env):
        self.env = env
        self.img = None

        # From boundingbox.yaml
        self.img_w = 1280
        self.img_h = 1280
        self.area_image = self.img_w * self.img_h

        self.time_tick = 1
        self.save_to_file = True
        self.save_dir = 'out/cam'
        self.bbox_dir = 'out/bbox'
        self.path_save = 'box'
        self.show_in_window = True
        self.filter_blocked = True
        self.max_distance = 1000
        self.min_distance = 1
        self.num_image = 1000

        self.include_vehicles = True
        self.include_walkers = True

        self.threshold_small_box = 3
        self.size_big_box = 70
        self.threshold_big_box = 10

        self.label_classes_name = {'car': 0, 'bus': 1, 'truck': 2, 'van': 3, 'walker': 4, 'street_light': 5}

        self.seg_color = {
            'car': [142, 0, 0],
            'truck': [142, 0, 0],
            'bus': [100, 60, 0],
            'van': [70, 0, 0],
            'bicycle': [32, 11, 119],
            'motorcycle': [230, 0, 0],
            'walker': [60, 20, 220],
            'street_light': [30, 170, 250],
        }

        self.vehicles_name = {
            'car': [
                'vehicle.audi.a2', 'vehicle.audi.etron', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer',
                'vehicle.chevrolet.impala', 'vehicle.citroen.c3', 'vehicle.dodge.charger_2020',
                'vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020', 'vehicle.ford.crown',
                'vehicle.ford.mustang', 'vehicle.jeep.wrangler_rubicon', 'vehicle.lincoln.mkz_2017',
                'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe', 'vehicle.mercedes.coupe_2020',
                'vehicle.micro.microlino', 'vehicle.mini.cooper_s', 'vehicle.mini.cooper_s_2021',
                'vehicle.nissan.micra', 'vehicle.nissan.patrol', 'vehicle.nissan.patrol_2021',
                'vehicle.seat.leon', 'vehicle.tesla.model3', 'vehicle.toyota.prius'
            ],
            'truck': [
                'vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv',
                'vehicle.carlamotors.firetruck', 'vehicle.tesla.cybertruck'
            ],
            'van': [
                'vehicle.ford.ambulance', 'vehicle.mercedes.sprinter',
                'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021'
            ],
            'bus': ['vehicle.mitsubishi.fusorosa'],
            'motorcycle': [
                'vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja',
                'vehicle.vespa.zx125', 'vehicle.yamaha.yzf'
            ],
            'bicycle': [
                'vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'
            ]
        }

    def IOU(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, (x2 - x1)) * max(0, (y2 - y1))
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection / float(box1_area + box2_area - intersection)
        return iou

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def get_image_point(self, loc, K, w2c):
        point = np.array([loc.x, loc.y, loc.z, 1])
        point_camera = np.dot(w2c, point)
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        point_img = np.dot(K, point_camera)
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
        return point_img[0:2]

    def calculate_visible_object(self, box1, box2, visibility_threshold=0.5):
        # Skip any invalid boxes with NaNs
        if any(np.isnan(box1)) or any(np.isnan(box2)):
            return False
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, (x2 - x1)) * max(0, (y2 - y1))
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if box1_area == 0 or box2_area == 0:
            return False
        visible_fraction = intersection / float(min(box1_area, box2_area))
        if visible_fraction >= visibility_threshold:
            return False
        print('not hidden')
        return True

    def convert_to_pillow(self, image):
        image.convert(self.env.carla.ColorConverter.Raw)
        i = np.array(image.raw_data)
        i2 = i.reshape((image.height, image.width, 4))
        i3 = np.zeros((image.height, image.width, 3))
        i3[:, :, 0] = i2[:, :, 2]
        i3[:, :, 1] = i2[:, :, 1]
        i3[:, :, 2] = i2[:, :, 0]
        i3 = np.uint8(i3)
        return Image.fromarray(i3, 'RGB')

    def retrieve_data(self, sensor_queue, frame, timeout=5):
        while True:
            try:
                data = sensor_queue.get(True, timeout)
            except queue.Empty:
                return None
            if data.frame == frame:
                return data

    def box_corrcttion(self, box):
        x_min, y_min, x_max, y_max = box
        if x_min >= self.img_w: return [0, 0, 0, 0]
        if x_min < 0: x_min = 0
        if y_min > self.img_h: return [0, 0, 0, 0]
        if y_min < 0: y_min = 0
        if x_max > self.img_w: x_max = self.img_w
        if x_max < 0: return [0, 0, 0, 0]
        if y_max > self.img_h: y_max = self.img_h
        if y_max < 0: return [0, 0, 0, 0]
        if (x_min >= x_max) or (y_min >= y_max): return [0, 0, 0, 0]
        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area == 0 or box_area >= self.area_image: return [0, 0, 0, 0]
        return [x_min, y_min, x_max, y_max]

    def filter_box(self, actor, box, img_seg):
        actor_type = getattr(actor, 'type_id', 'street_light')
        classification = self.get_actor_classification(actor_type)
        if classification is None: return False

        color_filter = self.seg_color[classification]
        x_min, y_min, x_max, y_max = box
        box_area = (x_max - x_min) * (y_max - y_min)
        img = img_seg[int(y_min):int(y_max), int(x_min):int(x_max), :3]
        match_pixels = np.sum(np.all(img == color_filter, axis=-1))

        pixel_ratio = (match_pixels / float(box_area)) * 100
        box_ratio = (box_area / float(self.area_image)) * 100

        if pixel_ratio <= self.threshold_small_box:
            return False
        if box_ratio >= self.size_big_box and pixel_ratio <= self.threshold_big_box:
            return False
        return True

    def filter_hidden_boxes(self, data):
        delete_boxes = []
        for n, actor in enumerate(data):
            box = actor['box']
            for i in range(n + 1, len(data)):
                second_box = data[i]['box']
                if not self.calculate_visible_object(box, second_box):
                    area1 = (box[2] - box[0]) * (box[3] - box[1])
                    area2 = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
                    delete_boxes.append(n if area1 < area2 else i)
        for i in sorted(set(delete_boxes), reverse=True):
            del data[i]
        return data

    def get_actor_classification(self, actor_type):
        for key, values in self.vehicles_name.items():
            if actor_type in values:
                return key
        if 'walker' in actor_type: return 'walker'
        if 'street_light' in actor_type: return 'street_light'
        return None

    def save_all_labels(self, data, path, name):
        """
        Save only pedestrian bounding boxes into a single YOLO label file,
        and return a list of label strings.
        """
        os.makedirs(path, exist_ok=True)
        label_path = os.path.join(path, name)

        labels = []
        with open(label_path, "w") as f:
            for d in data:
                if d['classification'] != 'walker':
                    continue  # Skip non-pedestrians

                box = copy.deepcopy(d['box'])  # [x1, y1, x2, y2]

                # Normalize
                box[0] /= float(self.img_w)
                box[1] /= float(self.img_h)
                box[2] /= float(self.img_w)
                box[3] /= float(self.img_h)

                # Convert to YOLO format
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                w = box[2] - box[0]
                h = box[3] - box[1]

                class_id = self.label_classes_name.get('walker', 4)
                label_str = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                f.write(label_str + "\n")
                labels.append(label_str)

        return labels

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

class Weather(object):
    def __init__(self, world, preset_index=7):
        self.weather_presets = [
            ("ClearNoon", carla.WeatherParameters.ClearNoon),
            ("CloudyNoon", carla.WeatherParameters.CloudyNoon),
            ("SoftRainNoon", carla.WeatherParameters.SoftRainNoon),
            ("HardRainNoon", carla.WeatherParameters.HardRainNoon),
            ("ClearNight", carla.WeatherParameters.ClearNight),
            ("CloudyNight", carla.WeatherParameters.CloudyNight),
            ("SoftRainNight", carla.WeatherParameters.SoftRainNight),
            ("HardRainNight", carla.WeatherParameters.HardRainNight),
        ]
        self.preset_name, self.weather = self.weather_presets[preset_index]

    def __str__(self):
        return self.preset_name

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args, metadata_writer=None, walker=None):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        self.metadata_writer = metadata_writer
        self.walker = walker
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            sys.exit(1)
        self.hud = hud
        self.intrinsic_matrix = np.array([
            [hud.dim[0] / (2.0 * math.tan(math.radians(90) / 2)), 0, hud.dim[0] / 2],
            [0, hud.dim[1] / (2.0 * math.tan(math.radians(90) / 2)), hud.dim[1] / 2],
            [0, 0, 1]
        ])
        self.player = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.lidar_sensor = None
        self.radar_sensor = None
        self.depth_camera = None
        self.semantic_lidar = None
        self.lidar_points = []
        self.radar_detections = []
        self.latest_seg = None
        self.bbox_generator = BoundingBox(self)
        self.bbox_generator.img_w = 1280
        self.bbox_generator.img_h = 1280
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

        self.yolo_label_dict = {}

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', 'hero')

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            spawn_point = carla.Transform(
                carla.Location(x=80.5, y=28.7, z=2.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player is not None:
                self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors after player is available
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self._spawn_sensors()


    def _spawn_sensors(self):
        blueprint_library = self.world.get_blueprint_library()

        rgb_bp = blueprint_library.find('sensor.camera.rgb')

        rgb_bp.set_attribute('fov', '90')
        rgb_bp.set_attribute('image_size_x', '1280')
        rgb_bp.set_attribute('image_size_y', '1280')
        rgb_bp.set_attribute('sensor_tick', '0.0')
        rgb_bp.set_attribute('shutter_speed', '200.0')
        rgb_bp.set_attribute('fstop', '1.4')
        rgb_bp.set_attribute('iso', '100.0')
        rgb_bp.set_attribute('gamma', '2.2')
        rgb_bp.set_attribute('bloom_intensity', '0.675')
        rgb_bp.set_attribute('lens_flare_intensity', '0.1')

        rgb_bp.set_attribute('lens_circle_falloff', '5.0')
        rgb_bp.set_attribute('lens_circle_multiplier', '0.0')
        rgb_bp.set_attribute('lens_k', '-1.0')
        rgb_bp.set_attribute('lens_kcube', '0.0')
        rgb_bp.set_attribute('lens_x_size', '0.08')
        rgb_bp.set_attribute('lens_y_size', '0.08')

        rgb_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
        self.rgb_camera = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.player)

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('fov', '90')
        depth_bp.set_attribute('image_size_x', '1280')
        depth_bp.set_attribute('image_size_y', '1280')
        depth_bp.set_attribute('sensor_tick', '0.0')
        depth_bp.set_attribute('lens_circle_falloff', '5.0')
        depth_bp.set_attribute('lens_circle_multiplier', '0.0')
        depth_bp.set_attribute('lens_k', '-1.0')
        depth_bp.set_attribute('lens_kcube', '0.0')
        depth_bp.set_attribute('lens_x_size', '0.08')
        depth_bp.set_attribute('lens_y_size', '0.08')
        depth_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.depth_camera = self.world.spawn_actor(depth_bp, depth_transform, attach_to=self.player)

        semantic_lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
        semantic_lidar_bp.set_attribute('channels', '64')
        semantic_lidar_bp.set_attribute('range', '50.0')
        semantic_lidar_bp.set_attribute('points_per_second', '100000')
        semantic_lidar_bp.set_attribute('rotation_frequency', '10.0')
        semantic_lidar_bp.set_attribute('upper_fov', '30.0')
        semantic_lidar_bp.set_attribute('lower_fov', '-40.0')
        semantic_lidar_bp.set_attribute('horizontal_fov', '360.0')
        semantic_lidar_bp.set_attribute('sensor_tick', '0.0')
        semantic_lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.semantic_lidar = self.world.spawn_actor(semantic_lidar_bp, semantic_lidar_transform, attach_to=self.player)

        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10.0')
        lidar_bp.set_attribute('upper_fov', '30.0')
        lidar_bp.set_attribute('lower_fov', '-40.0')
        lidar_bp.set_attribute('horizontal_fov', '360.0')
        lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')
        lidar_bp.set_attribute('dropoff_general_rate', '0.45')
        lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
        lidar_bp.set_attribute('sensor_tick', '0.0')
        lidar_bp.set_attribute('noise_stddev', '0.0')
        lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.player)

        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '60.0')
        radar_bp.set_attribute('vertical_fov', '5.0')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('sensor_tick', '0.0')
        radar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.radar_sensor = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.player)

        seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_cam_bp.set_attribute('image_size_x', '1280')
        seg_cam_bp.set_attribute('image_size_y', '1280')
        seg_cam_bp.set_attribute('fov', '90')
        seg_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.seg_camera = self.world.spawn_actor(seg_cam_bp, seg_transform, attach_to=self.player)

        # Optional: Save frames to disk
        weak_self = weakref.ref(self)
        self.rgb_camera.listen(lambda image: self._on_rgb_image(weak_self, image))
        self.depth_camera.listen(lambda image: self._on_depth_image(weak_self, image))
        self.semantic_lidar.listen(lambda data: self._on_semantic_lidar(weak_self, data))
        self.lidar_sensor.listen(lambda data: World._on_lidar_data(weak_self, data))
        self.radar_sensor.listen(lambda data: World._on_radar_data(weak_self, data))
        self.seg_camera.listen(lambda image: self._on_seg_image(weakref.ref(self), image))

    @staticmethod
    def _on_lidar_data(weak_self, lidar_data):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        if len(points) == 0:
            print("[LIDAR] No points received")
            return
        self.lidar_points = points
        try:
            clustering = DBSCAN(eps=0.75, min_samples=5, n_jobs=1).fit(points)
        except Exception as e:
            print(f"[LIDAR] DBSCAN failed: {e}")
            return
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[LIDAR] Detected {n_clusters} object clusters")

        for i in range(n_clusters):
            cluster = points[labels == i]
            centroid = np.mean(cluster, axis=0)
            for detection in self.radar_detections:
                dx = detection[0] - centroid[0]
                dy = detection[1] - centroid[1]
                dz = detection[2] - centroid[2]
                distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if distance < 2.0:
                    print(f"[FUSION] Moving object detected at {centroid} with radar velocity {detection[3]:.2f} m/s")

    @staticmethod
    def _on_radar_data(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        self.radar_detections = []
        for detection in radar_data:
            azimuth = detection.azimuth
            altitude = detection.altitude
            depth = detection.depth
            velocity = detection.velocity
            x = depth * math.cos(altitude) * math.cos(azimuth)
            y = depth * math.cos(altitude) * math.sin(azimuth)
            z = depth * math.sin(altitude)
            self.radar_detections.append((x, y, z, velocity))
            print(f'[RADAR] x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, velocity: {velocity:.2f}')

    def _enhance_with_sensor_fusion(self, box, actor, lidar_points, radar_detections, K, rgb_transform):
        actor_loc = actor.get_location()
        actor_pos = np.array([actor_loc.x, actor_loc.y, actor_loc.z])
        matched = False
        metadata = {'has_radar': False, 'radar_velocity': 0.0, 'lidar_points': 0}
        for radar_det in radar_detections:
            radar_pos = np.array(radar_det[:3])
            distance = np.linalg.norm(actor_pos - radar_pos)
            if distance < 2.0:
                metadata['has_radar'] = True
                metadata['radar_velocity'] = radar_det[3]
                matched = True
                break
        if not matched and len(lidar_points) > 0:
            dists = np.linalg.norm(lidar_points - actor_pos, axis=1)
            close_points = np.sum(dists < 2.0)
            if close_points > 0:
                matched = True
                metadata['lidar_points'] = int(close_points)
        return (box, 1.0 if matched else 0.5, metadata)

    def _on_rgb_image(self, weak_self, image):
        self = weak_self()
        if not self:
            return

        self.latest_rgb = image
        frame_id = image.frame

        # Get segmentation mask
        img_seg = None
        if hasattr(self, 'latest_seg') and self.latest_seg is not None:
            img_seg = np.frombuffer(self.latest_seg.raw_data, dtype=np.uint8).reshape(
                (self.latest_seg.height, self.latest_seg.width, 4))[:, :, :3].copy()

        # Collect actors
        actors = self.world.get_actors()
        walkers = [a for a in actors if 'walker.pedestrian' in a.type_id]

        # Camera transform and projection matrix
        rgb_transform = image.transform
        rgb_transform = np.array(rgb_transform.get_matrix())
        rgb_transform = np.linalg.inv(rgb_transform)
        K = self.bbox_generator.build_projection_matrix(self.bbox_generator.img_w, self.bbox_generator.img_h, 90.0)

        detected_objects = []
        cam_location = self.rgb_camera.get_transform().location
        current_lidar_points = getattr(self, 'lidar_points', np.empty((0, 3)))
        current_radar_detections = getattr(self, 'radar_detections', [])

        for actor in walkers:
            walker_location = actor.get_location()
            distance = walker_location.distance(cam_location)
            if distance < self.bbox_generator.min_distance or distance > self.bbox_generator.max_distance:
                continue
            bbox = actor.bounding_box
            extent = bbox.extent
            corners = [carla.Location(x=dx * extent.x, y=dy * extent.y, z=dz * extent.z)
                       for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]

            projected_points = []
            for corner in corners:
                world_corner = bbox.location + corner
                world_corner = actor.get_transform().transform(world_corner)
                try:
                    img_point = self.bbox_generator.get_image_point(world_corner, K, rgb_transform)
                    projected_points.append(img_point)
                except:
                    continue
            '''
            if len(projected_points) < 4:
                continue
            '''

            xs, ys = zip(*projected_points)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            corrected_box = self.bbox_generator.box_corrcttion([x_min, y_min, x_max, y_max])
            if corrected_box == [0, 0, 0, 0] or any(np.isnan(coord) for coord in corrected_box):
                continue

            # Filtering

            x_min, y_min, x_max, y_max = corrected_box
            w, h = x_max - x_min, y_max - y_min
            aspect_ratio = h / float(w) if w > 0 else 0
            if aspect_ratio > 12 or aspect_ratio < 0.25 or h < 5:
                continue

            box_area = w * h
            if box_area / float(self.bbox_generator.area_image) < 0.0005:
                continue

            # Optional semantic segmentation mask filter
            if img_seg is not None and not self.bbox_generator.filter_box(actor, corrected_box, img_seg):
                continue

            # Fusion confidence
            enhanced_box, confidence, metadata = self._enhance_with_sensor_fusion(
                corrected_box, actor, current_lidar_points, current_radar_detections, K, rgb_transform)

            if confidence >= 0.5:
                '''
                occluded = False
                for existing in detected_objects:
                    existing_box = existing['box']
                    if not self.calculate_visible_object(enhanced_box, existing_box):
                        occluded = True
                        break
                if occluded:
                    continue
                    '''
                detected_objects.append({
                    'box': enhanced_box,
                    'classification': 'walker',
                    'confidence': confidence,
                    'fusion_metadata': metadata,
                    'actor': actor
                })
            #detected_objects = self.filter_hidden_boxes(detected_objects)

        # Draw bounding boxes
        image_np = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :,
                   :3].copy()
        raw_image = os.path.expanduser(f'~/Downloads/images_raw/{frame_id:05d}.png')
        os.makedirs(os.path.dirname(raw_image), exist_ok=True)
        cv2.imwrite(raw_image, image_np)

        for i, d in enumerate(detected_objects):
            x_min, y_min, x_max, y_max = map(int, d['box'])
            color = (0, 255, 0) if d['confidence'] > 0.7 else (0, 255, 255) if d['confidence'] > 0.5 else (0, 0, 255)
            label = f"WALKER {i} ({d['confidence']:.2f})"
            metadata = d['fusion_metadata']
            if metadata.get('has_radar'):
                label += f" V:{metadata['radar_velocity']:.1f}"
            if metadata.get('lidar_points', 0) > 0:
                label += f" L:{metadata['lidar_points']}"
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, 3)


        # Save image
        save_path = os.path.expanduser(f"~/Downloads/images/{frame_id:05d}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_np)

        # Save labels
        label_path = os.path.expanduser(f"~/Downloads/labels/frame_{frame_id:06d}.txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        labels = self.bbox_generator.save_all_labels(detected_objects, os.path.dirname(label_path),
                                                     os.path.basename(label_path))
        self.yolo_label_dict[frame_id] = labels

        # Write metadata
        metadata_path = os.path.expanduser("~/Downloads/output_metadata.csv")
        write_header = not os.path.exists(metadata_path)

        with open(metadata_path, mode='a', newline='') as metadata_file:
            metadata_writer = csv.DictWriter(
                metadata_file,
                fieldnames=[
                    'frame', 'timestamp', 'weather', 'num_pedestrians',
                    'ego_x', 'ego_y', 'ego_z', 'ego_pitch', 'ego_yaw', 'ego_roll',
                    'cam_x', 'cam_y', 'cam_z', 'cam_pitch', 'cam_yaw', 'cam_roll',
                    'ego_speed_kmh', 'collision_type', 'lane_invasion', 'yolo_labels',
                    'walker_x', 'walker_y', 'walker_z', 'walker_speed_kmh', 'FOV'
                ]
            )


            if write_header:
                metadata_writer.writeheader()

            tf = self.player.get_transform()
            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
            cam_tf = self.camera_manager.sensor.get_transform() if (
                    self.camera_manager and self.camera_manager.sensor
            ) else self.rgb_camera.get_transform()
            label_map = {v: k for k, v in self.bbox_generator.label_classes_name.items()}
            if hasattr(self, 'walker') and self.walker is not None:
                walker_location = self.walker.get_location()
                walker_velocity = self.walker.get_velocity()
                walker_speed_kmh = 3.6 * math.sqrt(walker_velocity.x ** 2 + walker_velocity.y ** 2 + walker_velocity.z ** 2)
            else:
                walker_location = carla.Location()
                walker_speed_kmh = 0.0

            # Write metadata row
            metadata_writer.writerow({
                'frame': frame_id,
                'timestamp': self.hud.simulation_time,
                'weather': str(self.hud._weather_sim),
                'num_pedestrians': len(pedestrians),
                'ego_x': tf.location.x,
                'ego_y': tf.location.y,
                'ego_z': tf.location.z,
                'ego_pitch': tf.rotation.pitch,
                'ego_yaw': tf.rotation.yaw,
                'ego_roll': tf.rotation.roll,
                'cam_x': cam_tf.location.x,
                'cam_y': cam_tf.location.y,
                'cam_z': cam_tf.location.z,
                'cam_pitch': cam_tf.rotation.pitch,
                'cam_yaw': cam_tf.rotation.yaw,
                'cam_roll': cam_tf.rotation.roll,
                'ego_speed_kmh': self.hud.speed,
                'collision_type': self.hud.collision_type,
                'lane_invasion': self.hud.lane_invasion,
                'yolo_labels': '|'.join([f"{label_map[int(l.split()[0])]}:{l}" for l in labels]),
                'walker_x': walker_location.x,
                'walker_y': walker_location.y,
                'walker_z': walker_location.z,
                'walker_speed_kmh': walker_speed_kmh,
                'FOV': 90
            })

    def _on_depth_image(self, weak_self, image):
        self = weak_self()
        if not self:
            return
        self.latest_depth = image

    def _on_semantic_lidar(self, weak_self, data):
        self = weak_self()
        if not self:
            return
        self.latest_semantic_lidar = data

    def _on_seg_image(self, weak_self, image):
        self = weak_self()
        if not self:
            return
        self.latest_seg = image


    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        if hasattr(self, 'latest_rgb'):
            array = np.frombuffer(self.latest_rgb.raw_data, dtype=np.uint8)
            array = np.reshape(array, (self.latest_rgb.height, self.latest_rgb.width, 4))
            surface = pygame.surfarray.make_surface(array[:, :, :3].swapaxes(0, 1))
            display.blit(surface, (0, 0))
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.rgb_camera,
            self.depth_camera,
            self.seg_camera,
            self.semantic_lidar,
            self.lidar_sensor,
            self.radar_sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player
        ]
        for actor in actors:
            if actor is not None:
                try:
                    if hasattr(actor, 'is_listening') and actor.is_listening:
                        actor.stop()
                    actor.destroy()
                except RuntimeError:
                    print(f"[Warning] Actor {getattr(actor, 'id', 'unknown')} already destroyed or not listening.")

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height, weather_sim):
        """Constructor method"""
        self.dim = (width, height)
        self._weather_sim = weather_sim
        self.speed = 0.0
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.collision_type = "None"
        self.lane_invasion = "None"
        self._collision_frame = -1
        self._lane_frame = -1
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.collision_duration = 0

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        self.speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        pedestrians = world.world.get_actors().filter('walker.pedestrian.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % self.speed,
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            'Weather: % 20s' % str(self._weather_sim),
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        other_actor = event.other_actor
        actor_type = other_actor.type_id

        # Compute impulse magnitude
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        # Add to collision history
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.popleft()

        # === PEDESTRIAN COLLISION CHECK ===
        if "walker.pedestrian" in actor_type or "ragdoll" in actor_type.lower():
            print(f"[Collision] PEDESTRIAN HIT! Actor type: {actor_type}")
            self.hud.notification("PEDESTRIAN COLLISION DETECTED!", seconds=2.5)
        else:
            print(f"[Collision] Actor: {actor_type}, Impulse: {intensity:.2f}")

        self.hud.notification("Collision with %r" % actor_type)
        self.hud.collision_type = event.other_actor.type_id
        self.hud._collision_frame = event.frame

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))
        self.hud.lane_invasion = ' and '.join(text)
        self.hud._lane_frame = event.frame

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self.latest_image = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.queue_rgb = queue.Queue()
        self.queue_seg = queue.Queue()
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=1.6, y=0.0, z=1.5), carla.Rotation(pitch=-15)), attachment.Rigid)
]
        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.sensors[self.index][0] == 'sensor.camera.rgb':
            self.latest_image = image
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    pygame.init()
    world = None
    background_vehicles = []

    try:
        if args.seed is not None:
            print(f"[Seed] Using random seed: {args.seed}")
            import random as pyrandom
            pyrandom.seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        weather_sim = Weather(sim_world.get_weather())
        sim_world.set_weather(weather_sim.weather)
        hud = HUD(args.width, args.height, weather_sim)
        # Create world (World will handle metadata internally)
        for actor in sim_world.get_actors().filter('walker.pedestrian.*'):
            actor.destroy()
        for actor in sim_world.get_actors().filter('controller.ai.walker'):
            actor.destroy()


        # Spawn walker
        bp_library = sim_world.get_blueprint_library()
        walker_bp = bp_library.find('walker.pedestrian.0001')
        controller_bp = bp_library.find('controller.ai.walker')
        walker_bp.set_attribute('is_invincible', 'false')

        walker_spawn = carla.Transform(carla.Location(x=114, y=2, z=1.0))
        walker_target = carla.Location(x=100, y=2, z=1.0)

        walker = sim_world.try_spawn_actor(walker_bp, walker_spawn)
        walker_controller = None
        delay_frames = 0
        delay_counter = 0
        delay_started = False
        if walker:
            walker.set_simulate_physics(True)
            walker_controller = sim_world.try_spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker)
            if walker_controller:
                walker_controller.start()
                walker_controller.set_max_speed(1.0)
                delay_started = True
            else:
                walker.destroy()
                print("[ERROR] Walker controller failed to spawn.")
        else:
            print("[ERROR] Failed to spawn walker.")

        world = World(client.get_world(), hud, args, walker = walker)
        world.collision_sensor = CollisionSensor(world.player, hud)
        world.lane_invasion_sensor = LaneInvasionSensor(world.player, hud)
        controller = KeyboardControl(world)
        # Setup agent
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 25)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Spawn background vehicles
        vehicle_bp = bp_library.filter('vehicle.*')
        vehicle_spawn_points = world.map.get_spawn_points()
        random.shuffle(vehicle_spawn_points)
        num_vehicles = 0
        for i in range(num_vehicles):
            if i >= len(vehicle_spawn_points):
                break
            blueprint = random.choice(vehicle_bp)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color',
                    random.choice(blueprint.get_attribute('color').recommended_values))
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.world.try_spawn_actor(blueprint, vehicle_spawn_points[i])
            if vehicle:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.ignore_lights_percentage(vehicle, 0.0)
                traffic_manager.ignore_signs_percentage(vehicle, 0.0)
                traffic_manager.vehicle_percentage_speed_difference(
                    vehicle, random.randint(-20, 10))
                background_vehicles.append(vehicle)

        # Destination for agent
        destination = carla.Location(
            x=random.uniform(106.1, 106.3),
            y=random.uniform(-5.5, -4.5),
            z=0.0)
        agent.set_destination(destination)

        clock = pygame.time.Clock()

        # ================= Main Simulation Loop =================
        while True:
            clock.tick()
            current_frame = world.hud.frame
            if current_frame - world.hud._collision_frame > 3:
                world.hud.collision_type = ""
            if current_frame - world.hud._lane_frame > 3:
                world.hud.lane_invasion = ""
            if world.hud.collision_type != "":
                world.hud.collision_duration += 1
            else:
                world.hud.collision_duration = 0

            if world.hud.collision_duration > 6:
                print("[STOP] Collision lasted more than 6 frames. Ending simulation.")
                break

            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()

            if controller.parse_events():
                return

            world.tick(clock)
            sim_world.set_weather(world.hud._weather_sim.weather)
            world.render(display)
            pygame.display.flip()

            if delay_started and delay_counter == delay_frames:
                walker_controller.go_to_location(walker_target)
                delay_started = False
            elif delay_started:
                delay_counter += 1

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(vehicle_spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("Target reached, searching for another.")
                else:
                    print("Target reached, stopping simulation.")
                    break

            try:
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
            except RuntimeError:
                print("[Warning] Agent step failed or actor destroyed.")
                break

    finally:
        if world and world.lidar_sensor:
            try:
                world.lidar_sensor.stop()
            except RuntimeError:
                pass
        if world and world.radar_sensor:
            try:
                world.radar_sensor.stop()
            except RuntimeError:
                pass
        if world:
            try:
                world.destroy()
            except RuntimeError:
                pass
        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x1280',
        help='Window resolution (default: 1280x1280)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
