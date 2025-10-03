#!/usr/bin/env python
import os
import sys
import rospy
import threading
import random
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry

class UAVControl:
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.state_sub = rospy.Subscriber(f'/uav{uav_id}/mavros/state', State, self.state_cb)
        self.odom_sub = rospy.Subscriber(f'/uav{uav_id}/mavros/local_position/odom', Odometry, self.odom_cb)
        self.local_pos_pub = rospy.Publisher(f'/uav{uav_id}/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy(f'/uav{uav_id}/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy(f'/uav{uav_id}/mavros/set_mode', SetMode)
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.target_pose = PoseStamped()
        self.rate = rospy.Rate(50)  # 50 Hz
        self.setpoint_thread = None
        self.target_reached = False
        self.target_pose.pose.position.x = 0.0
        self.target_pose.pose.position.y = 0.0
        self.target_pose.pose.position.z = 40.0  # Initial takeoff height

    def state_cb(self, state):
        self.current_state = state

    def odom_cb(self, odom):
        self.current_pose.pose = odom.pose.pose
        # Check if target reached (within 0.5m)
        dx = self.current_pose.pose.position.x - self.target_pose.pose.position.x
        dy = self.current_pose.pose.position.y - self.target_pose.pose.position.y
        dz = self.current_pose.pose.position.z - self.target_pose.pose.position.z
        if (dx**2 + dy**2 + dz**2)**0.5 < 0.5:
            self.target_reached = True

    def publish_setpoint(self):
        while not rospy.is_shutdown():
            self.target_pose.header.frame_id = "map"
            self.target_pose.header.stamp = rospy.Time.now()
            self.target_pose.pose.orientation.w = 1.0
            self.local_pos_pub.publish(self.target_pose)
            if self.target_reached:
                with threading.Lock():
                    self.target_pose.pose.position.x = random.uniform(-70, 70)
                    self.target_pose.pose.position.y = random.uniform(-70, 70)
                    self.target_pose.pose.position.z = random.uniform(25, 120)
                    self.target_reached = False
                    rospy.loginfo(f"UAV{self.uav_id} new setpoint: x={self.target_pose.pose.position.x:.2f}, "
                                  f"y={self.target_pose.pose.position.y:.2f}, z={self.target_pose.pose.position.z:.2f}")
            self.rate.sleep()

    def control(self):
        # Wait for FCU connection
        while not self.current_state.connected and not rospy.is_shutdown():
            self.rate.sleep()

        # Start setpoint publishing
        self.setpoint_thread = threading.Thread(target=self.publish_setpoint)
        self.setpoint_thread.daemon = True
        self.setpoint_thread.start()

        # Wait briefly
        rospy.sleep(1)

        # Arm UAV
        rospy.wait_for_service(f'/uav{self.uav_id}/mavros/cmd/arming')
        try:
            self.arming_client(True)
        except rospy.ServiceException as e:
            rospy.logerr(f"UAV{self.uav_id} arming failed: {e}")

        # Switch to OFFBOARD mode
        rospy.wait_for_service(f'/uav{self.uav_id}/mavros/set_mode')
        try:
            self.set_mode_client(custom_mode="OFFBOARD")
        except rospy.ServiceException as e:
            rospy.logerr(f"UAV{self.uav_id} set mode failed: {e}")

def get_uav_count():
    topics = rospy.get_published_topics()
    uav_ids = set()
    for topic, _ in topics:
        if topic.startswith('/uav') and 'mavros/state' in topic:
            uav_id = int(topic.split('/')[1].replace('uav', ''))
            uav_ids.add(uav_id)
    return sorted(list(uav_ids))

def main(argv=sys.argv):
    try:
        rospy.init_node('swarm_control', anonymous=True)
        uav_ids = get_uav_count()
        rospy.loginfo(f"Detected {len(uav_ids)} UAVs: {uav_ids}")
        controllers = []
        for uav_id in uav_ids:
            controller = UAVControl(uav_id)
            controllers.append(controller)
            threading.Thread(target=controller.control, daemon=True).start()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass