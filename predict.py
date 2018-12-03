import shutil
import random
import os
import numpy as np
import cv2
import argparse
import sys
import time

from scripts.label_image import *
from vid_frame import *
from mode_accuracy import *

def run(video_path, frame_dir):

	segment_vid(video_path, frame_dir)

	predicted_word = predict_on_vid(frame_dir, model_file, label_file, input_height, input_width, input_mean, input_std, input_layer, output_layer, graph, input_name, output_name, input_operation, output_operation, sample_size)

	return predicted_word

if __name__ == "__main__":
	file_name = ""
	direct_name = "segmented_frames/"
	model_file = "tf_files/retrained_graph.pb"
	label_file = "tf_files/retrained_labels.txt"
	#Uncomment 'model_file' and 'label_file' below to switch to 64 word model:
	#model_file = "tf_files/retrained_graph2.pb"
	#label_file = "tf_files/retrained_labels2.txt"
	input_height = 224
	input_width = 224
	input_mean = 128
	input_std = 128
	input_layer = "input"
	output_layer = "final_result"

	input_name = "import/" + input_layer
	output_name = "import/" + output_layer

	graph = load_graph(model_file)
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);

	sample_size = 20 # number of images to randomly sample from the processed video

	parser = argparse.ArgumentParser()
	parser.add_argument("--video", help="video to be processed")
	parser.add_argument("--directory", help="directory in which processed frames are saved")
	parser.add_argument("--graph", help="graph/model to be executed")
	parser.add_argument("--labels", help="name of file containing labels")
	parser.add_argument("--input_height", type=int, help="input height")
	parser.add_argument("--input_width", type=int, help="input width")
	parser.add_argument("--input_mean", type=int, help="input mean")
	parser.add_argument("--input_std", type=int, help="input std")
	parser.add_argument("--input_layer", help="name of input layer")
	parser.add_argument("--output_layer", help="name of output layer")
	args = parser.parse_args()

	if args.graph:
		model_file = args.graph
	if args.video:
		file_name = args.video
	if args.directory:
		direct_name = args.directory
	if args.labels:
		label_file = args.labels
	if args.input_height:
		input_height = args.input_height
	if args.input_width:
		input_width = args.input_width
	if args.input_mean:
		input_mean = args.input_mean
	if args.input_std:
		input_std = args.input_std
	if args.input_layer:
		input_layer = args.input_layer
	if args.output_layer:
		output_layer = args.output_layer

	predicted_word = run(file_name, direct_name)

	#remove segmented frames from directory in case you're 
	#running multiple videos sequentially 
	shutil.rmtree(direct_name)
	os.mkdir(direct_name)








