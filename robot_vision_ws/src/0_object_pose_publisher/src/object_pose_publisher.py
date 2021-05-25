#!/bin/python3 

#tensorflow for model
import tensorflow as tnsf
from tensorflow.python.ops.gen_array_ops import parallel_concat_eager_fallback
#ros for python
import rospy
import tf
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

#custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'customModel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'datasetPreparation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classes', 'positionSolver'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from featureModel import featureModel
from residualModel import residualModel
from positionSolver import positionSolver

from datasetPreparation import datasetPreparation

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg, args):
    #unpack args
    child = args[0]
    parent = args[1] 
    model = args[2]
    ps_prediction = args[3]

    try:
        # Convert your ROS Image message to OpenCV2
        #img = bridge.imgmsg_to_cv2(msg, "bgr8")
        img = bridge.imgmsg_to_cv2(msg, "rgb8")
        #resize the image
        img = cv2.resize(img, (400,400))
        #save image
        cv2.imwrite('camera_image.jpeg', img)
        #convert from numpy to tensor
        img = tnsf.convert_to_tensor(img, dtype = tnsf.float32)
        #add batch dimension for model
        img = tnsf.expand_dims(img, axis=0)
        #get predictions
        logits_beliefs, logits_affinities = model(img, training=False,)
        #model prediciton
        rot_pred, tran_pred, test_img, _ = ps_prediction.getPosition(logits_beliefs, logits_affinities, img)

        #print predictions (translation is divided by 1000 because it is in [m] and transform needs to be in [mm])
        tran_pred = tran_pred/1000
        tnsf.print("predicted translation is:")
        {tnsf.print("\t\t\t{},".format(tran)) for tran in tran_pred}
        tnsf.print("predicted rotation is:")
        {tnsf.print("\t\t\t{},".format(rot)) for rot in rot_pred}

        br = tf.TransformBroadcaster()
        #publishing transform between child and parent
        print("Publishing transform between [{}] --> [{}]".format(parent, child))
        br.sendTransform((tran_pred[0], tran_pred[1], -tran_pred[2]),
                        tf.transformations.quaternion_from_euler(rot_pred[0], rot_pred[1], rot_pred[2]),
                        rospy.Time.now(),
                        child,
                        parent)
        #save image
        test_img = Img.fromarray(test_img)
        test_img.save(os.path.join( "", 'test.png'))
    except CvBridgeError as e:
        print(e)

def listener(model, node_name):

    ####only for debug####
    #sub_topic="/fanuc_1/fixed_camera_pcl/image_raw"
    #child="thor_hammer"
    #parent="fixed_camera_link"
    #camsettings = "/home/ros/Desktop/tensorflow_model/camera_object_settings/_camera_settings.json"
    #objsettings = "/home/ros/Desktop/tensorflow_model/camera_object_settings/_object_settings.json"

    #get parameters from parameter server
    camsettings = rospy.get_param("{}/camera_settings".format(node_name))
    objsettings = rospy.get_param("{}/object_settings".format(node_name))
    child = rospy.get_param("{}/child".format(node_name))
    parent = rospy.get_param("{}/parent".format(node_name))
    sub_topic = rospy.get_param("{}/camera_topic".format(node_name))
    
    #print parameters
    print("camera settings: {}".format(camsettings))
    print("object settings: {}".format(objsettings))
    print("child name: {}".format(child))
    print("parent name: {}".format(parent))
    print("camera topic: {}".format(sub_topic))

    ps_prediction = positionSolver(camsettings, objsettings, True, text_width_ratio=0.01, text_height_ratio=0.1, 
                                    text = 'Logit',  belColor = (0, 255, 0), affColor = (0, 255, 0)) 

    # Define your image subscriber
    rospy.Subscriber(sub_topic, Image, image_callback, (child, parent, model, ps_prediction), queue_size=1, buff_size=2**24)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    """
    if(opt.model=='featureModel'):
        tf.print('Creating feature model')
        netModel = featureModel(pretrained=True, blocks=6, numFeatures=512, freezeLayers=14,)
    elif(opt.model=='residualModel'):
        tf.print('Creating residual model')
        netModel = residualModel(pretrained=True, blocks=6, freezeLayers=14,)
    """

    #init node
    rospy.init_node('object_pose_publisher', anonymous=True)
    #print node name
    print("Node name: {}".format(rospy.get_name()))
    node_name = rospy.get_name()
    #print param list
    #print("Param names: {}".format(rospy.get_param_names()))

    #get model class from parameter server
    model_name = rospy.get_param("{}/model_name".format(node_name))
    #create model
    netModel = featureModel(pretrained=True, blocks=6, numFeatures=512, freezeLayers=14,)
    if(model_name=='featureModel'):
        tnsf.print('Creating feature model')
        netModel = featureModel(pretrained=True, blocks=6, numFeatures=512, freezeLayers=14,)
    elif(model_name=='residualModel'):
        tnsf.print('Creating residual model')
        netModel = residualModel(pretrained=True, blocks=6, freezeLayers=14,)
    # model can be built by calling the build function but then all of the layers have to be used.
    # or by calling the fit function
    # to load weights model has to be built
    tnsf.print('building model: {}'.format(netModel.name))
    netModel.build(input_shape=(None, 400, 400, 3))

    #get checkpoint path from parameter server
    ckptpath = rospy.get_param("{}/ckptpath".format(node_name))
    tnsf.print('loading weights from: {}'.format(ckptpath))
    #load weight from checkpoint
    netModel.load_weights(filepath=ckptpath)

    listener(netModel, node_name)