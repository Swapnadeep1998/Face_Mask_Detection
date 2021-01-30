import tensorflow as tf 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils 
from object_detection.builders import model_builder 
from object_detection.utils import config_util 
import os 


model=tf.saved_model.load('./InferenceGraph/saved_model')

@tf.function
def detection_fn(image):    
    model_fn = model.signatures['serving_default']
    detections = model_fn(image)   
    return detections

