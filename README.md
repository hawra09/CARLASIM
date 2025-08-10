The purpose of this repo is to house the information and scripts necessary to carry out my capstone project. 
The project focuses on improving pedestrian detection in preset CARLA weather conditions such as: [ClearNoon, CloudyNoon, SoftRainNoon, HardRainNoon, ClearNight, CloudyNight, SoftRainNight, HardRainNight]. 
The base script is the automatic_control.py file from CARLA Python API example folder. To that script several features were added:
  - Preset Weather Conditions
  - Unified walker scenario: the walker spawns at defined location (x=114, y=2. z=1.0) and target (x=100, y=2, z=1.0)
  - Many sensors have been incorporated and activated.
        -RGB: serves as the primary sensor for pedestrian detection
        -Segementation: serves to filter detections to ensure higher accuracy, precision, and precision. Speficially, uses the fiter_boxes() method from the bounding box class. **At the moment, this sensor is problematic because it is causing detection to essentially become non-existent after the two initial frames.**
        - LiDAR and RADAR:Generally, RADAR is better suited in long-distance situations and is less affected by weather conditions. Specifically, RADAR uses radio frequency travel time to measure distance. Whereas, LiDAR uses well-suited in short-distance situations and is more affected by weather conditions. LiDAR measures distance by using laser pulses. Therefore, sensor fusion enhances detection and confidence. DBSCAN clustering method is used for the point cloud data to detect movements and clusters.
       - Collision Sensor: Modifications include terminating the simulation if collision is longer than six frames.
       - GNSS Sensor: No modifications were done to this particular sensor. The purpose of this sensor is provide longtitude and latitude metadata related to the ego vehicle.
       - Lane Invasion Sensor: No modifications were done to this particular sensor. The purpose is to detect lane boundary crossing.
   - The depth and semantic LiDAR sensors were implemented, but they were not used.
   - Bounding Box:
        -This adapts code from:  https://github.com/Mofeed-Chaar/Improving-bouning-box-in-Carla-simulator/tree/main
                -Main functions of this class include: calculating the intrinsic camera matrix, projects 3D-bounding boxes into 2D plane, assigns segmentation color for each class [car, truck, bus, van, bicycle, motorcycle, walker, and street_light], calculates the IOU (intersection over union), calculates the visibility threshold for objects, prevents negative coordinates meaning that object location will not be behind the sensor and/or the camera (
  - Set the ego vehicle to a Telsa Model 3
  - Shifted the camera
