
TRACK

Simple Object Tracking based on detect.py from https://github.com/google-coral/examples-camera/blob/master/gstreamer/detect.py
and unmodified sort.py tracking algorithm from https://github.com/abewley/sort/blob/master/sort.py

see https://github.com/abewley/sort for algorithm details and link to original paper


# run tracker on camera input
python3 track.py --model all_models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite --labels all_models/coco_labels.txt 

