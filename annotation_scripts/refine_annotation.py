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
    # print(videoInfo)

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

def extractFrames(videoPath, suffix_str, ss, duration, fps):
    ffmpeg.input(videoPath, ss=ss, to=ss+duration).filter('fps', fps).output(f'{videoPath}_{suffix_str}_frame%05d.jpg').run()

def refineSegments(videoPath, bStart = True):
    # load video segments from corresponding json file
    videoSegs = loadVideoJson(videoPath)
    print(videoSegs)

    for segId, seg in enumerate(videoSegs):
        fps = 10
        duration = 4

        ss = seg['start_time']
        if seg['2nd_serve'] > 0.0:
            ss = seg['2nd_serve']

        if not bStart:
            ss = seg['end_time']

        extractFrames(videoPath, f"seg{segId:02d}", ss - duration / 2, duration, fps)
        # print(ss)

"""
Extract frames based on segType specification
segType:
  0: return the whole video as one segment
  1: return all the non-point segments
  2: return all the in-point segments
"""
def extractSegmentFrames(videoPath, segType, fps = 0):
    videoInfo = probeVideoInfo(videoPath)
    print(videoInfo)

    segs = []

    # Return whole video as one segment (if segType == 0)
    if (segType == 0) or (segType > 2): 
        segs.append([0, videoInfo['duration']])

    if (segType == 1) or (segType == 2):
        # load video segments from corresponding json file
        videoSegs = loadVideoJson(videoPath)
        # print(videoSegs)
        if len(videoSegs) <= 0:
            return None
        
        # add each segments (non-point or in_point)
        ss = 0
        duration = 0    
        for id, seg in enumerate(videoSegs):
            # set ss/duration for non-point segment
            duration = seg['start_time'] - ss

            # set ss/duration for in-point segment
            if segType == 2:
                ss = seg['start_time']
                # replace start time with 2nd_serve's timestamp
                ss_2nd = seg['2nd_serve']
                if ss_2nd > 0.0:
                    ss = ss_2nd

                # calculate duration
                duration = seg['end_time'] - ss
                if (duration < 1): # quit if something wrong: e.g., duration should not be too short
                    return None
            # add a segment
            segs.append([ss, duration])

            # update start_time for next non-point segment
            ss = seg['end_time']

        # need to add one more segment for non-point segments (segment after last point ended)
        if segType == 1:
            duration = videoInfo['duration'] - ss
            if duration > 1:
                segs.append([ss, duration])

    # extract frames if fps > 0
    suffix_arr = ["", "none", "point"]
    if fps > 0:
        # extracting frames
        for segId, gap in enumerate(segs):
            extractFrames(videoPath, f"{suffix_arr[segType]}_{segId:02d}", gap[0], gap[1], fps)

    # update stats
    if (segType == 1) or (segType == 2):
        sum = 0
        for seg in segs:
            sum += seg[1]  # sum of all duration
        print(f"Percentage of {suffix_arr[segType]} is ", sum / videoInfo['duration'])
    
    return segs        
    
if __name__ == '__main__':
    # Parse argument to set "CourtId" (default to 0): 
    #     if CourtId = 0 --> book by time. 
    #     if CourtId != 0 --> load from config file (_BookCourt.txt)
    parser = argparse.ArgumentParser()
    parser.add_argument('CourtId', type=int, nargs='?', default=1, help="If 1: book by court number. If 0, book by time")
    args = parser.parse_args()

    # hard code the video segment to test
    videoFn = "00092_seg2.mov"
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    videoPath = os.path.join(scriptDir, "tmp", videoFn)
    # print(videoPath)
    # videoInfo = probeVideoInfo(videoPath)
    # print(videoInfo)

    # load video segments from corresponding json file
    videoSegs = loadVideoJson(videoPath)
    # print(videoSegs)

    # Refine json segments
    # refineSegments(videoPath, True)
    # refineSegments(videoPath, False)

    # extract frames for testing or training
    gaps = extractSegmentFrames(videoPath, segType=2, fps=0)
    # print(gaps)