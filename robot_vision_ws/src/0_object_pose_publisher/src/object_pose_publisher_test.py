#!/bin/python3 

#tensorflow for model
import tensorflow as tnsf
from tensorflow.python.ops.gen_array_ops import parallel_concat_eager_fallback
#ros for python
import rospy
import tf2_ros
import tf
from tf import transformations
import geometry_msgs.msg
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
#system
import sys
import os
#PIL
from PIL import Image as Img
import numpy as np
# Import math Library
import math


#custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'customModel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'datasetPreparation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'positionSolver'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from featureModel import featureModel
from residualModel import residualModel
from positionSolver import positionSolver


# int getQuaternion(const cv::Mat_<double> &rvec, cv::Mat_<double> &quat) {
#     cv::Mat_<double> R(3, 3);
#     cv::Rodrigues(rvec, R);

#     if ((quat.rows == 4) && (quat.cols == 1)) {
#         //Mat size OK
#     } else {
#         quat = cv::Mat_<double>::eye(4,1);
#     }
#     double   w;

#     w = R(0,0) + R(1,1)+ R(2,2) + 1;
#     if ( w < 0.0 ) return 1;

#     w = sqrt( w );
#     quat(0,0) = (R(2,1) - R(1,2)) / (w*2.0);
#     quat(1,0) = (R(0,2) - R(2,0)) / (w*2.0);
#     quat(2,0) = (R(1,0) - R(0,1)) / (w*2.0);
#     quat(3,0) = w / 2.0;
#     return 0;
# }


def getQuaternion(rvec):

    quat = np.zeros((4,1), np.float)

    R, _ = cv2.Rodrigues(rvec)
    print(R)

    # if ((quat.shape[0] == 4) and (quat.shape[1] == 1)):
    #     quat = quat
    # else:
    #     quat = np.zeros((4,1), np.float)

    w = R[0][0] + R[1][1] + R[2][2] + 1

    if (w<0.0):
        return quat

    w = math.sqrt(w)
    quat[0,0] = (R[2,1] - R[1,2]) / (w*2.0)
    quat[1,0] = (R[0,2] - R[2,0]) / (w*2.0)
    quat[2,0] = (R[1,0] - R[0,1]) / (w*2.0)
    quat[3,0] = w / 2.0

    return quat



def getGeometryMsgTransform(parent, child, tran_pred, euler_angles, rot_matrix):
    
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = tran_pred[0]
    t.transform.translation.y = tran_pred[1]
    t.transform.translation.z = tran_pred[2]

    if (rot_matrix is None) and (euler_angles is not None):
        quat = tf.transformations.quaternion_from_euler(euler_angles[0], euler_angles[1], euler_angles[2])

    if (rot_matrix is not None) and (euler_angles is None):
        quat = tf.transformations.quaternion_from_matrix(rot_matrix)

    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    print("euler from quaternion:")
    print(tf.transformations.euler_from_quaternion(quat))

    return t

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg, args):
    #unpack args
    child = args[0]
    parent = args[1] 
    #model = args[2]
    ps_prediction = args[2]
    debug_image = args[3]

    try:
        # Convert your ROS Image message to OpenCV2
        #img = bridge.imgmsg_to_cv2(msg, "bgr8")
        img = bridge.imgmsg_to_cv2(msg, "rgb8")
        #mirror image 
        #img = cv2.flip(img, 1)
        #resize the image
        #img = cv2.resize(img, (400,400))
        #save image
        cv2.imwrite('camera_image.jpeg', img)
        #convert from numpy to tensor
        img = tnsf.convert_to_tensor(img, dtype = tnsf.float32)
        #add batch dimension for model
        img = tnsf.expand_dims(img, axis=0)
        #get predictions
        #logits_beliefs, logits_affinities = model(img, training=False,)
        #model prediciton
        #rot_pred, tran_pred, test_img, _ = ps_prediction.getPosition(logits_beliefs, logits_affinities, img)

        #print predictions (translation is divided by 1000 because it is in [m] and transform needs to be in [mm])
        tran_pred = np.array([[0.08500429], [0.06712567], [-0.72157958]])
        print (" tran pred shape: {}".format(tran_pred.shape))
        rot_pred = np.array([[-0.96650259], [-0.74452698], [-1.07668671]])
        print (" rot_pred: shape {}".format(rot_pred.shape))
        #tran_pred = tran_pred/1000
        tran_pred[0] = -tran_pred[0]
        tran_pred[1] = -tran_pred[1]
        tran_pred[2] = -tran_pred[2]

        tnsf.print("predicted translation is:")
        {tnsf.print("\t\t\t{},".format(tran)) for tran in tran_pred}
        tnsf.print("predicted rotation is:")
        {tnsf.print("\t\t\t{},".format(rot)) for rot in rot_pred}

        rotation_matrix = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1]],
                                    dtype=float)

        #rot_pred =  np.array([rot_pred[0], rot_pred[1], rot_pred[2]])
        #rotation_matrix[:3, :3], _ = cv2.Rodrigues(rot_pred)

        #quat = tf.transformations.quaternion_from_matrix(rotation_matrix)
        #euler = tf.transformations.euler_from_quaternion(quat)
        #print("Euler rotation before is: {} ".format(euler))
        #euler = np.array([euler[0], euler[1], euler[2]])
        #print("Euler rotation after is: {} ".format(euler))
        #print("Quaternion from Euler is: {} ".format(tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])))
        #print("Quaternion from Rotation matrix is: {} ".format(tf.transformations.quaternion_from_matrix(rotation_matrix)))

        #1 solution not working
        rot, _ = cv2.Rodrigues(rot_pred)
        
        #rot = np.linalg.inv(rot)
        #rotation_matrix[:3, :3] = rot

        #projection matrix
        P = np.hstack((rot,tran_pred))
        euler_angles = cv2.decomposeProjectionMatrix(P)[6]
        euler_angles=np.radians([euler_angles[0], euler_angles[1], euler_angles[2]])
        print ("Euler angles in radians: {}".format(euler_angles))

        #camera position
        pos_cam = -np.matrix(rot).T * np.matrix(tran_pred)
        print ("Camera position: {}".format(pos_cam))

        #R = transformations.euler_matrix(0, 0, 0)
        #R = np.matrix(rot).T
        #roll, pitch, yaw = transformations.euler_from_matrix(R)
        #euler_angles[0] = roll
        #euler_angles[1] = pitch
        #euler_angles[2] = yaw
        euler_angles[0] = euler_angles[0] #-math.pi
        euler_angles[1] = euler_angles[1]
        euler_angles[2] = euler_angles[2]
        # euler_angles[0] = rot_pred[0]
        # euler_angles[1] = rot_pred[1]
        # euler_angles[2] = rot_pred[2]       

        #create transform for broadcaster
        t = getGeometryMsgTransform(parent, child, tran_pred, euler_angles, None)
        # t = getGeometryMsgTransform(parent, child, tran_pred, None, rotation_matrix)

        br = tf2_ros.TransformBroadcaster()

        #publishing transform between child and parent
        print("Publishing transform between [{}] --> [{}]".format(parent, child))
        br.sendTransform(t)

        if debug_image:
            #get current dir path
            dir_path = os.path.dirname(os.path.realpath(__file__))
            img_path = os.path.join( dir_path, 'test.png')
            #save image
            #print('Saving image on path: {}'.format(img_path))
            #test_img = Img.fromarray(test_img)
            #test_img.save(os.path.join( img_path))

    except CvBridgeError as e:
        print(e)

def listener(node_name):

    ####only for debug####
    #sub_topic="/fanuc_1/fixed_camera_pcl/image_raw"
    #child="thor"
    #parent="camera"
    #camsettings = "/home/ros/Desktop/robot_vision_ws/src/0_object_pose_publisher/json/cam_settings.json"
    #objsettings = "/home/ros/Desktop/robot_vision_ws/src/0_object_pose_publisher/json/_object_settings.json"
    #debug_image = 'True'

    #get parameters from parameter server
    camsettings = rospy.get_param("{}/camera_settings".format(node_name))
    objsettings = rospy.get_param("{}/object_settings".format(node_name))
    child = rospy.get_param("{}/child".format(node_name))
    parent = rospy.get_param("{}/parent".format(node_name))
    sub_topic = rospy.get_param("{}/camera_topic".format(node_name))
    debug_image = rospy.get_param("{}/debugImage".format(node_name))
    
    #print parameters
    print("camera settings: {}".format(camsettings))
    print("object settings: {}".format(objsettings))
    print("child name: {}".format(child))
    print("parent name: {}".format(parent))
    print("camera topic: {}".format(sub_topic))
    print("debug image: {}".format(debug_image))

    ps_prediction = positionSolver(None, camsettings, True, objsettings, True, text_width_ratio=0.01, text_height_ratio=0.1, 
                                    text = 'Logit',  belColor = (0, 255, 0), affColor = (0, 255, 0)) 

    # Define your image subscriber
    rospy.Subscriber(sub_topic, Image, image_callback, (child, parent, ps_prediction, debug_image), queue_size=1, buff_size=2**24)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':

    #init node
    rospy.init_node('object_pose_publisher', anonymous=True)
    #print node name
    print("Node name: {}".format(rospy.get_name()))
    node_name = rospy.get_name()

    #model_name = 'featureModel'
    #get model class from parameter server
    #model_name = rospy.get_param("{}/model_name".format(node_name))
    #create model
    #netModel = featureModel(pretrained=True, blocks=6, numFeatures=512, freezeLayers=14,)
    #if(model_name=='featureModel'):
    #    tnsf.print('Creating feature model')
    #    netModel = featureModel(pretrained=True, blocks=6, numFeatures=512, freezeLayers=14,)
    #elif(model_name=='residualModel'):
    #    tnsf.print('Creating residual model')
    #    netModel = residualModel(pretrained=True, blocks=6, freezeLayers=14,)
    # model can be built by calling the build function but then all of the layers have to be used.
    # or by calling the fit function
    # to load weights model has to be built
    #tnsf.print('building model: {}'.format(netModel.name))
    #netModel.build(input_shape=(None, 400, 400, 3))

    #get checkpoint path from parameter server
    #ckptpath = '/home/ros/Desktop/Tensorflow_model/ckpt/testModel_blocks_6/cp.ckpt'
    #ckptpath = rospy.get_param("{}/ckptpath".format(node_name))
    #tnsf.print('loading weights from: {}'.format(ckptpath))
    #load weight from checkpoint
    #netModel.load_weights(filepath=ckptpath)

    listener( node_name)