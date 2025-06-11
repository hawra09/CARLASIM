The purpose of this repo is to house the information and scripts necessary to carry out my capstone project. 
The project focuses on improving pedestrian detection in dynamic weather conditions. 
The base script is the automatic_control.py file from CARLA Python API example folder. To that script several features were added:
  - Dynamic weather
  - Spawn pedestrians
  - Spawn traffic vehicles
  - Sensors: liDAR and radar. Uses DBSCAN from sklearn for object clustering.
  - Set the ego vehicle to a Telsa Model 3
  - Shifted the camera
