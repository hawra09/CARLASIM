The purpose of this repo is to house the information and scripts necessary to carry out my capstone project. 
The project focuses on improving pedestrian detection in preset CARLA weather conditions such as: [ClearNoon, CloudyNoon, SoftRainNoon, HardRainNoon, ClearNight, CloudyNight, SoftRainNight, HardRainNight]. 
The base script is the automatic_control.py file from CARLA Python API example folder. To that script several features were added:
  - Preset Weather Conditions
  - Unified walker scenario: the walker spawns at defined location (x=114, y=2. z=1.0) and target (x=100, y=2, z=1.0)
  - Many sensors have been incorporated and activated. The ego vehicle spawn locations are initial spawn point (x=80.5, y=28.7, z=2.0) and target is a small, range: x = [106.1, 106.3], y = [-5.5, -4.5], z = [0]
        -RGB: serves as the primary sensor for pedestrian detection. Futhermore, *geometric filtering was applied to reduce the number of false positives.* Future studies should focus on more advace filtering mechanisms such as temporal tracking, this will help to address concerns regarding motion dynamics which causes mislocalized false negatives. *Reducing the number of false negatives is critical for autonomous vehicle related perception systems.*
        -Segementation: serves to filter detections to ensure higher accuracy, precision, and precision. Speficially, uses the fiter_boxes() method from the bounding box class. **At the moment, this sensor is problematic because it is causing detection to essentially become non-existent after the two initial frames.**
        - LiDAR and RADAR:Generally, RADAR is better suited in long-distance situations and is less affected by weather conditions. Specifically, RADAR uses radio frequency travel time to measure distance. Whereas, LiDAR uses well-suited in short-distance situations and is more affected by weather conditions. LiDAR measures distance by using laser pulses. Therefore, sensor fusion enhances detection and confidence. DBSCAN clustering method is used for the point cloud data to detect movements and clusters.
       - Collision Sensor: Modifications include terminating the simulation if collision is longer than six frames.
       - GNSS Sensor: No modifications were done to this particular sensor. The purpose of this sensor is provide longtitude and latitude metadata related to the ego vehicle.
       - Lane Invasion Sensor: No modifications were done to this particular sensor. The purpose is to detect lane boundary crossing.
   - The depth and semantic LiDAR sensors were implemented, but they were not used.
   - Bounding Box:
        -This adapts code from:  https://github.com/Mofeed-Chaar/Improving-bouning-box-in-Carla-simulator/tree/main
                -Main functions of this class include: calculating the intrinsic camera matrix, projects 3D-bounding boxes into 2D plane, assigns segmentation color for each class [car, truck, bus, van, bicycle, motorcycle, walker, and street_light] which is used for semantic filtering (several thresholds are set in place to trigger this filtering mechanism), calculates the IOU (intersection over union), calculates the visibility threshold for objects, prevents negative coordinates meaning that object location will not be behind the sensor and/or the camera (also known as phantom detections), and finally bounding boxes are normalized converted to YOLO-format. 
  - Most importantly, metadata collection has been implemented and attributes such as: [frame, timestamp, weather, num_pedestrians, ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_roll, cam_x, cam_y, cam_z, cam_pitch, cam_yaw, cam_roll, ego_speed_kmh, collision_type, lane_invasion, yolo_labels, walker_x, walker_y, walker_z, walker_speed_kmh, FOV].
  - For easy debugging and analysis, two sets of images are being stored per simulation: raw_RGB images and labelled_RGB images (bounding boxes are drawn onto the image via CV2).
  - Shifted the camera, so it's at the driver's eye level.
  - 
