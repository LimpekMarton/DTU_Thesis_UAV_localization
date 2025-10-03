import os
import sys
import threading
from copy import deepcopy
from collections import namedtuple
from collections import deque as cdeque
from queue import deque, Empty
import pickle
import argparse
import datetime
import pdb
import itertools
import csv
import math

import rospy
from gazebo_nova_ros_plugin.msg import VectorConnection
from std_msgs.msg import String
from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped


conndata = []
rxmovedata = []

last_msg = None
num_rx = None
num_tr = None
recievers = None
transmitters = None
con_glob = None
done = False
current_position = {}
imu_data = {}
local_pose = {}
clock_time = None  # Global simulation time
file_locks = {}

class ConnectClass:
	def __init__(self, tr_id = None, rx_id = None, path_loss = None):
		self.tr_id = tr_id
		self.rx_id = rx_id
		self.path_loss = path_loss
		
	def __str__(self):
		return 'Transmitter_id: ' + str(self.tr_id) + '\nReceiver_id: ' + str(self.rx_id) + '\nPath_loss (in db): ' + str(self.path_loss) +'\n\n'
		
	def getTrId(self):
		return self.tr_id
		
	def getRxId(self):
		return self.rx_id
	
	def getPathLoss(self):
		return float(self.path_loss)
		

def state_callback(msg):
    global done
    if msg.data == "done":
        print("DONE")
        done=True
    
def connections_callback(msg):
	global con_glob, num_rx, num_tr, transmitters, recievers, done
	#if num_rx is None:
	num_rx = len(msg.recievers)
	num_tr = len(msg.transmitters)
	con_glob = msg.connections
		#print(con_glob)
	t = rospy.Time.now()





def findBestConForRec(rec_id, connections_list):
	"""Find the best connection path in connections_list for a receiver"""
	min_path_loss = 500
	tr_id = 0
	for i in range(0, len(connections_list), 1):
		if rec_id == connections_list[i].getRxId() and min_path_loss > connections_list[i].getPathLoss():
			min_path_loss = connections_list[i].getPathLoss()
			tr_id = connections_list[i].getTrId()
	return (tr_id, min_path_loss)
	
	
def findAllConForRec(rec_id, connections_list):
	"""Find all paths in connections_list for a receiver"""
	all_conn_for_rec = []
	for i in range(0, len(connections_list), 1):
		if connections_list[i].getRxId() == rec_id and connections_list[i].getPathLoss() < 400:
			#print(connections_list[i])
			all_conn_for_rec.append(connections_list[i].getPathLoss())
	#print(all_conn_for_rec[1])
	return all_conn_for_rec

def findBestConTransRec(rec_id, connections_list, number_of_uavs):
	"""This function finds the best connections from the transmitters given in the connections_list
	to a given receiver (rec_id)"""

	best_connections = []
	tr_ids = []
	for i in range(0, len(connections_list), 1):
		if connections_list[i].getRxId() == rec_id and connections_list[i].getPathLoss() < 500 and connections_list[i].getTrId() not in tr_ids:
			tr_ids.append(connections_list[i].getTrId())
	#print(f'rec_id: {rec_id}, TR ids: {tr_ids}')

	#Now we now all the TR ids, which is connected to rec_id
	#As for the next step, lets find the minimum pathloss for each rec-tx id pairs
	#there is one receiver which is connected to multiple transmitters and we need to find the best paths
	#which means minimum pathloss
	min_losses = {k: 500 for k in range(number_of_uavs)} #lets say the min path_loss is 500 from all transmitter to receiver
	for conn in connections_list:
		if rec_id == conn.getRxId() and conn.getPathLoss() < min_losses[conn.getTrId()]:
			min_losses[conn.getTrId()] = conn.getPathLoss()

	
	for trans_id in min_losses.keys():
		if min_losses[trans_id]==500:
			best_connections.append(ConnectClass(tr_id=trans_id, rx_id=rec_id, path_loss=math.nan))
		else:
			best_connections.append(ConnectClass(tr_id=trans_id, rx_id=rec_id, path_loss=min_losses[trans_id]))

	return best_connections
	

def make_one_step():
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause_physics()
        rospy.sleep(0.1)  # Adjust the delay as needed
        rospy.wait_for_service('/gazebo/pause_physics')
        pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pause_physics()
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", str(e))



def gps_callback(msg, uav_id):
    global current_position
    try:
        current_position[uav_id] = (msg.latitude, msg.longitude, msg.altitude)
    except Exception as e:
        rospy.logerr(f"UAV{uav_id} GPS error: {e}")


def imu_callback(data, uav_id):
    global imu_data
    imu_data[uav_id] = [data.angular_velocity, data.linear_acceleration] #TODO: remove orientation - DONE
    #rospy.loginfo(f"UAV{uav_id} IMU: Orientation=[{data.orientation.x}, {data.orientation.y}, {data.orientation.z}, {data.orientation.w}]")

def pose_callback(data, uav_id):
    global local_pose
    local_pose[uav_id] = [data.pose.orientation, data.pose.position] #TODO: get oriantation from here
    #rospy.loginfo(f"UAV{uav_id} Pose: Local=[{data.pose.position.x}, {data.pose.position.y}, {data.pose.position.z}]")

def clock_callback(data):
    global clock_time
    clock_time = data.clock.to_sec()
    #rospy.loginfo(f"Clock: {clock_time.to_sec()}")


# Thread function for each UAV
def uav_data_writer(uav_id, csv_file_path, num_transmitters, number_of_uavs):
    global con_glob, num_tr, num_rx, last_msg, recievers, transmitters, done, imu_data, current_position, local_pose, clock_time, file_locks
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown() and not done:
        local_pathlosses = []
        imu_values = []
        orientation_and_pose_loc = []
        try:
            if uav_id in current_position and num_rx is not None:
                connections_local = []
                con_loc = con_glob
                #print(f'UAV_id: {uav_id}, Global connections: {con_loc}')
                for i in range(len(con_loc)):
                    connections_local.append(ConnectClass(con_loc[i].from_, con_loc[i].to, con_loc[i].db_loss))
                #print(f'UAV_id: {uav_id}, Connections local: {connections_local[0]}')
                best_for_each = findBestConTransRec(uav_id, connections_local, number_of_uavs)
                #print(f'UAV_id: {uav_id}, Best: {best_for_each}')
                sorted_connections = sorted(best_for_each, key=lambda best_for_each: best_for_each.tr_id)

                #print(sorted_connections)

                # Printing the data
                for co in sorted_connections:
                    local_pathlosses.append(co.path_loss)

                #print(local_pathlosses)
                #print(f'ID: {uav_id}, clock: {clock_time}')
                #print(f'ID: {uav_id}, IMU: {imu_data[uav_id]}')
                #print(f'ID: {uav_id}, Local pose: {local_pose[uav_id]}')
                #print(f'ID: {uav_id}, GPS: {current_position[uav_id]}')

                clk = [clock_time]

                imu_values = [
                    imu_data[uav_id][0].x, imu_data[uav_id][0].y, imu_data[uav_id][0].z,
                    imu_data[uav_id][1].x, imu_data[uav_id][1].y, imu_data[uav_id][1].z
                ]

                orientation_and_pose_loc = [
                    local_pose[uav_id][0].x, local_pose[uav_id][0].y, local_pose[uav_id][0].z, local_pose[uav_id][0].w,
                    local_pose[uav_id][1].x, local_pose[uav_id][1].y, local_pose[uav_id][1].z
                ]


                position_values = list(current_position[uav_id])

                row_data = local_pathlosses + clk + imu_values + orientation_and_pose_loc + position_values
                with file_locks[uav_id]:
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow(row_data)
                        csvfile.flush()

            rate.sleep()

        except Exception as e:
            rospy.logerr(f"ERROR: {e}")



def collect_and_print_data():
    """Generates dataset for each UAV using threads"""
    global con_glob, num_tr, num_rx, last_msg, recievers, transmitters, done, imu_data, current_position, local_pose, clock_time, file_locks

    # Initialize ROS node
    rospy.init_node('uav_swarm_data_collector', anonymous=True)

    # Get number of UAVs
    number_of_uavs = len([t for t in rospy.get_published_topics() if t[0].startswith('/uav') and 'mavros/global_position/global' in t[0]])
    num_transmitters = num_tr
    #rospy.loginfo(f"Number of UAVs: {number_of_uavs}, Transmitters: {num_transmitters}")

    # Initialize subscribers
    connection_sub = rospy.Subscriber('/raytrace/ray_connections', VectorConnection, connections_callback)
    clock_sub = rospy.Subscriber('/clock', Clock, clock_callback)
    drone_pos_sub = []
    drone_imu_sub = []
    drone_local_pose = []
    #file_locks = {}
	
    # Create CSV files and locks for each UAV
    for uav_id in range(number_of_uavs):
        file_locks[uav_id] = threading.Lock()
        csv_file_path = f'/home/remote/ariac_ws/uav{uav_id}_data.csv'
        # Dynamic headers: path losses for all other UAVs as transmitters, if i != uav_id
        headers = [f'Transmitter_{i}_path_loss' for i in range(number_of_uavs)] + ['Time',
        'Angular_Velocity_X', 'Angular_Velocity_Y', 'Angular_Velocity_Z',
        'Linear_Acceleration_X', 'Linear_Acceleration_Y', 'Linear_Acceleration_Z',
        'Orientation_X', 'Orientation_Y', 'Orientation_Z', 'Orientation_W',
        'Local_X', 'Local_Y', 'Local_Z',
        'Latitude', 'Longitude', 'Altitude',
    ]
        # Initialize CSV with headers
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(headers)

        # Subscribe to UAV-specific topics
        try:
            drone_pos_sub.append(rospy.Subscriber(f'/uav{uav_id}/mavros/global_position/global', NavSatFix, lambda msg, id=uav_id: gps_callback(msg, id)))
            drone_imu_sub.append(rospy.Subscriber(f'/uav{uav_id}/mavros/imu/data_raw', Imu, lambda msg, id=uav_id: imu_callback(msg, id)))
            drone_local_pose.append(rospy.Subscriber(f'/uav{uav_id}/mavros/local_position/pose', PoseStamped, lambda msg, id=uav_id: pose_callback(msg, id)))
        except Exception as e:
            rospy.logerr(f"Fail in ros topic subscription: {e}")

    #rospy.sleep(0.1)
    #print(current_position)
    #print("-----------------")
    #print(imu_data)
    #print("-----------------")
    #print(local_pose)
    #print("-----------------")
    #print(clock_time)

    


    # Start a thread for each UAV
    threads = []
    for uav_id in range(number_of_uavs):
        csv_file_path = f'/home/remote/ariac_ws/uav{uav_id}_data.csv'
        thread = threading.Thread(target=uav_data_writer, args=(uav_id, csv_file_path, num_transmitters, number_of_uavs))
        thread.daemon = True
        threads.append(thread)
        thread.start()

    # Keep node running until shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        done = True
        rospy.loginfo("Shutting down")
    finally:
        done = True
        for thread in threads:
            thread.join()
	




def main(argv=sys.argv):
    global con_glob, num_tr, num_rx, last_msg,recievers, transmitters, done, imu_data, current_position
    connections_local = []
    connections_in_class = []
    myargv = rospy.myargv(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', default='')
    args = parser.parse_args(myargv[1:])

    collect_and_print_data()

    return 0
