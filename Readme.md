## Face Mask Detection Using TFOD API Tensorflow 2.x

1. Clone models repo
    >> git clone https://github.com/tensorflow/models.git

2. Compile protos files (inside research folder)
    >> protoc object_detection/protos/*.proto --python_out=.

3. Copy tf2 setup.py file inside research directory
    >> cp object_detection/packages/tf2/setup.py .

4. Install dependencies
    >> python3 -m pip install .

5. Test the installations
    >> python3 object_detection/builders/model_builder_tf2_test.py

6. Place your dataset inside object_detection folder, and convert XML files to csv
    >> python3 xml_to_csv.py

7. Convert CSV files to TFReords
   >> python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

   >> python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

8. Create folder inside object_detection with name 'training'. Inside it create a file 'label_map.pbtxt'

9. Edit the model_pipeline_config file

11. Start the training 
    >> python3 model_main_tf2.py --pipeline_config_path=training/pipeline.config --model_dir=training --alsologtostderr

12. Export the inference graph
    >> python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/pipeline.config --trained_checkpoint_dir training/ --output_directory InferenceGraph/



Note. On paperspace
    >> sudo apt update
    >> sudo apt install libgl1-mesa-glx
    
    
    
Credit: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
