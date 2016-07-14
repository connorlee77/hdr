hdr_cal
=======

This tool calibrates an mvBlueFOX-x00w HDR imaging sensor. Clone this into the `src` directory of your ROS workspace, and make sure to also have mv_camera in there.

 - `roscore`
 - `rosrun mv_camera mv_camera_node`
 - `rosrun hdr_cal cal.py`

 The tool will then find a good exposure level in the current environment (where very few pixels are saturated), and will then calculate the optimal HDR parameters given the histogram of the image at the ideal exposure. It will set the parameters to the camera, and will also print them to the terminal.
