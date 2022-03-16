# ECE276A-PR2


The data needs to be downloaded as explained in the project writeout and added in the data folder. 

To run the program, simply run 

`python3 slam.py`

Files: 

slam.py: Main file that runs the whole program. 

particle_filter.py: File that contains prediction and update step.

mapping.py: File that contains the map update function.

sync_fog_to_encoder.py: File for synchronizing the fog data to the encoder time stamps and creating omegas.

dead_reckoning.py: File for creating dead reckoning trajectory to check prediction step.

pr2_utils.py: Utilities file. 
