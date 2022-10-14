#! /usr/bin/env python
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose



def main():
    rospy.init_node('get_pos_in_map_frame')
    tflistener = tf.TransformListener()
    tflistener.waitForTransform('/map','/base_footprint',rospy.Time(),rospy.Duration(4.0))
    try:
         (trans,rot) = tflistener.lookupTransform('/map','/base_footprint',rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr('Error occure when transfrom.')
        return 
    print('                [%f,%f,%f],[%f,%f,%f,%f]     ' % (trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]))
 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo(" node terminated.")  


