import os, sys
import json
import argparse

import math
from PIL import Image, ImageOps
import numpy as np

import glob
import ffmpeg
from datetime import datetime

def probeVideoInfo(videoPath):
    if (not os.path.exists(videoPath)):
        print("File does not exist")
        return

    res = ffmpeg.probe(videoPath, cmd='ffprobe')
    # print(res)
    res = res['streams'][0]

    info = {}
    info['duration'] = res['duration']
    info['width'] = res['width']
    info['height'] = res['height']
    info['frame_rate'] = eval(res['r_frame_rate'])
    info['avg_frame_rate'] = eval(res['avg_frame_rate'])
    
    info['creation_time'] = datetime.today().strftime('%Y-%m-%d')
    info['rotated'] = False
    if ('tags' in res):
        if ('creation_time' in res['tags']):
            info['creation_time'] = res['tags']['creation_time'][:10]

        if ('rotate' in res['tags']):
            if (res['tags']['rotate'] == 90) or (res['tags']['rotate'] == 270):
                info['rotated'] = True

    return info

def loadVideoJson(videoPath):
    videoJsonPath = videoPath + '.json'
    if not os.path.exists(videoJsonPath):
        return None
    else:            
        with open(videoJsonPath, 'r') as json_file:
            dict = json.load(json_file)
    return dict

def extractFrames(videoPath, segId, startTime, numFrames, fps):
    ffmpeg.input(videoPath, ss=startTime).filter('fps', fps).output(f'{videoPath}_seg{segId:02d}_frame%03d.jpg', vframes=numFrames).run()
    
def refineSegments(videoPath, bStart = True):
    # load video segments from corresponding json file
    videoSegs = loadVideoJson(videoPath)
    print(videoSegs)

    segId = 0
    for seg in videoSegs:
        if seg['in_point']:
            fps = 10
            ss = seg['end_time']
            numFrames = 8 * fps
            if bStart:
                ss = seg['start_time']
                numFrames = 4 * fps            
            halfT = numFrames / fps // 2
            extractFrames(videoFn, segId, ss - halfT, numFrames, fps)
            # print(ss)
            segId += 1

if __name__ == '__main__':
    # Parse argument to set "CourtId" (default to 0): 
    #     if CourtId = 0 --> book by time. 
    #     if CourtId != 0 --> load from config file (_BookCourt.txt)
    parser = argparse.ArgumentParser()
    parser.add_argument('CourtId', type=int, nargs='?', default=1, help="If 1: book by court number. If 0, book by time")
    args = parser.parse_args()

    # hard code the video segment to test
    videoFn = "tmp//seg3.mov"
    videoInfo = probeVideoInfo(videoFn)
    print(videoInfo)

    # load video segments from corresponding json file
    videoSegs = loadVideoJson(videoFn)
    print(videoSegs)

    # Refine json segments
    # refineSegments(videoFn, True)
    refineSegments(videoFn, False)
