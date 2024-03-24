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
    info['duration'] = float(res['duration'])
    info['width'] = int(res['width'])
    info['height'] = int(res['height'])
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
    if not os.path.exists(videoPath):
        return None
    
    videoInfo = probeVideoInfo(videoPath)
    print(videoInfo)

    with open(videoJsonPath, 'r') as json_file:
        dict = json.load(json_file)
    
    # check if annotation is valid (number of entries > 0)
    if (len(dict) <= 0):
        return None

    # check if video length is consistent with annotation
    time_dif = abs(dict[0]['video_len'] - videoInfo['duration'])
    if (time_dif > 1):
        return None

    return dict

def extractFrames(videoPath, segId, startTime, numFrames, fps):
    ffmpeg.input(videoPath, ss=startTime).filter('fps', fps).output(f'{videoPath}_seg{segId:02d}_frame%03d.jpg', vframes=numFrames).run()
    
def refineSegments(videoPath, bStart = True):
    # load video segments from corresponding json file
    videoSegs = loadVideoJson(videoPath)
    print(videoSegs)

    segId = 0
    for seg in videoSegs:
        fps = 10
        numFrames = 4 * fps

        ss = seg['start_time']
        if seg['2nd_serve'] > 0.0:
            ss = seg['2nd_serve']

        if not bStart:
            ss = seg['end_time']

        halfT = numFrames / fps // 2
        extractFrames(videoPath, segId, ss - halfT, numFrames, fps)
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
    videoFn = "20240312_195520_seg3.mov"
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    videoPath = os.path.join(scriptDir, "tmp", videoFn)
    print(videoPath)
    # videoInfo = probeVideoInfo(videoPath)
    # print(videoInfo)

    # load video segments from corresponding json file
    videoSegs = loadVideoJson(videoPath)
    print(videoSegs)

    # Refine json segments
    # refineSegments(videoPath, True)
    # refineSegments(videoPath, False)
