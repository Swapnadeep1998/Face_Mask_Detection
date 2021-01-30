import tensorflow as tf 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils 
from object_detection.builders import model_builder 
from object_detection.utils import config_util 
import os 

CONFIG_PATH='/home/swapnadeep/Documents/Machine_learning_projects/Face_Mask_Detection/Building/training/pipeline.config'
CHECKPOINT_PATH = '/home/swapnadeep/Documents/Machine_learning_projects/Face_Mask_Detection/Building/training'
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'],is_training=False)

# ckpt = tf.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join(CHECKPOINT_PATH,'ckpt-66')).expect_partial()

model=tf.saved_model.load('./InferenceGraph/saved_model')

@tf.function
def detection_fn(image):    
    model_fn = model.signatures['serving_default']
    detections = model_fn(image)   
    return detections

