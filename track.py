# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Simple Object Tracking based on detect.py from https://github.com/google-coral/examples-camera/blob/master/gstreamer/detect.py
#	and unmodified sort.py tracking algorithm from https://github.com/abewley/sort/blob/master/sort.py
#
#	see https://github.com/abewley/sort for algorithm details and link to original paper
#
#
#
#

import argparse
import collections
import common
import gstreamer
import numpy as np
import os
import re
import svgwrite
import time

#from tkinter import *	
from sort import *	#tracker algorithm

from filterpy.kalman import KalmanFilter

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x+1, y+1), fill='black', font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill='white', font_size=font_size))
"""
def generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines):
    dwg = svgwrite.Drawing('', size=src_size)
    src_w, src_h = src_size
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        shadow_text(dwg, 10, y*20, line)
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)        # Relative coordinates.
        x, y, w, h = x0, y0, x1 - x0, y1 - y0        # Absolute coordinates, input tensor space.
        x, y, w, h = int(x * inf_w), int(y * inf_h), int(w * inf_w), int(h * inf_h)        # Subtract boxing offset.
        x, y = x - box_x, y - box_y        # Scale to source coordinate space.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        percent = int(100 * obj.score)
#        label = '{}% equals {}'.format(percent, labels.get(obj.id, obj.id))
        label = 'obj id:{}'.format(percent)
        shadow_text(dwg, x, y - 5, label)
        dwg.add(dwg.rect(insert=(x,y), size=(w, h),
                        fill='none', stroke='red', stroke_width='2'))
    return dwg.tostring()
"""

# ##############################################################################################################
def generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines,trdata):
    dwg = svgwrite.Drawing('', size=src_size)
    src_w, src_h = src_size
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h
    #    print('objs: ',objs)
    #    print('track: ',trdata)
    for y, line in enumerate(text_lines, start=1):
        shadow_text(dwg, 10, y*20, line)
    obj=objs[0]
    for td in trdata:
        x0,y0,x1,y1=td[0].item(),td[1].item(),td[2].item(),td[3].item()
        print('track id: {}   {} {} {} {}'.format(td[4],x0,x1,y0,y1))
        overlap=0
        for ob in objs:
            dx0,dy0,dx1,dy1=ob.bbox.xmin.item(),ob.bbox.ymin.item(),ob.bbox.xmax.item(),ob.bbox.ymax.item()
            area=(min(dx1,x1)-max(dx0,x0))*(min(dy1,y1)-max(dy0,y0))
            if (area>overlap):
                overlap=area
                obj=ob
            print('    {:6.3f} {:6.3f} {:6.3f} {:6.3f} area={:6.3f} {:8s} {:3d}%   best: {:8s} {:3d}% '
                .format(dx0,dy0,dx1,dy1,area,labels.get(ob.id, ob.id),
                int(100*ob.score),labels.get(obj.id,obj.id),int(100*obj.score)))
        x, y, w, h = x0, y0, x1 - x0, y1 - y0      # Relative coordinates.    
        x, y, w, h = int(x * inf_w), int(y * inf_h), int(w * inf_w), int(h * inf_h) # Absolute coordinates
        x, y = x - box_x, y - box_y           # Subtract boxing offset.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y          # Scale to source coordinate space.
        percent = int(100 * obj.score)
        label = 'id:{:3.0f}  {} {}%'.format(td[4], labels.get(obj.id, obj.id),percent)
        shadow_text(dwg, x, y - 5, label)
        dwg.add(dwg.rect(insert=(x,y), size=(w, h),fill='none', stroke='red', stroke_width='2'))
    return dwg.tostring()
# ##############################################################################################################
       
class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    category_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def main():
    nframe=1
    tracker=Sort()   #create instance of SORT tracker
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    w, h, _ = common.input_image_size(interpreter)
    inference_size = (w, h)
    # Average fps over last 30 frames.
    fps_counter  = common.avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box):
      nonlocal tracker
      nonlocal nframe
      nonlocal fps_counter
      start_time = time.monotonic()
      common.set_input(interpreter, input_tensor)
      interpreter.invoke()
      # For larger input image sizes, use the edgetpu.classification.engine for better performance
      objs = get_output(interpreter, args.threshold, args.top_k)
      end_time=time.monotonic()

      dets= [] #np.array([])
      print('num objs: ',len(objs))
      for n in range (0,len(objs)):
        element=[] # np.array([])
##        print('object id ',n)
#        print(objs[n].bbox.xmin,objs[n].bbox.ymin,objs[n].bbox.xmax,objs[n].bbox.ymax,objs[n].score)
        element.append(objs[n].bbox.xmin)
        element.append(objs[n].bbox.ymin)
        element.append(objs[n].bbox.xmax)
        element.append(objs[n].bbox.ymax)
        element.append(objs[n].score)   #    print('element= ',element)
        dets.append(element)    #      print('dets: ',dets)
      dets=np.array(dets)       #convert to numpy array #      print('npdets: ',dets)

      trdata=tracker.update(dets)
      track_end=time.monotonic()    #      print('tracker: ',trdata)
      text_lines = [
          'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
          'Track: {:.2f} ms'.format((track_end-end_time)*1000),
          'FPS: {} fps'.format(round(next(fps_counter))),
      ]
      print('nframe=',nframe,' '.join(text_lines))
      nframe+=1
      return generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines,trdata)

    result = gstreamer.run_pipeline(user_callback,src_size=(640, 480),appsink_size=inference_size,
            videosrc=args.videosrc,videofmt=args.videofmt)

if __name__ == '__main__':
    main()


