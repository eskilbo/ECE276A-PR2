# ECE276A PR3 Visual-Inertial SLAM

The data needs to be downloaded as explained in the project writeout and added in the data folder.

To run the program, simply run

`python3 main.py`

Files:

main.py: Main file that runs the whole program, starting with the prediction and mapping and ending with the VI SLAM. 

slam.py: File that contains the class for EKF Slam for use in task c.

mapping.py: File that contains the class EKF Mapping for use for prediction and visual mapping (tasks a and b)

pr2_utils.py: Utilities file including new helper functions for mapping and slam as well as the visualization function.
