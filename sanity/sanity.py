#!/usr/bin/python3.7
import json
import os
import re
import cv2


def dump_dict(filepath, data):
    with open(filepath, "w") as fd:
        json.dump(data, fd, sort_keys=True, indent=4)


def get_line_info(line):
    line_info = None
    m = re.search("^([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)", line)
    if m:
        frame_num = int(m.group(1))
        person_id = int(m.group(2))
        xmin = int(m.group(3))
        ymin = int(m.group(4))
        width = int(m.group(5))
        height = int(m.group(6))
        line_info = (frame_num, person_id, xmin, ymin, width, height)
    return(line_info)


def get_anno_info(anno_file):
    with open(anno_file, "r") as fd:
        content = fd.readlines()
    anno_info = {}
    for line in content:
        line_info = get_line_info(line)
        if line_info != None:
            frame_num, person_id, xmin, ymin, width, height = line_info
            if frame_num not in anno_info:
                anno_info[frame_num] = []
            person_tuple = (person_id, xmin, ymin, width, height)
            anno_info[frame_num].append(person_tuple)
    return(anno_info)


def get_person_color(person_id):
    person_num = 90
    chan_num = 3
    chan_size = 255
    step = (chan_num * chan_size) / (person_num)
    raw_color = int(person_id * step)
    person_color = [0, 0, 0]
    for chan_index in range(0, chan_num):
        if raw_color < ((chan_index + 1) * chan_size):
            chan_color = raw_color - chan_index * chan_size
            person_color[chan_index] = chan_color
            break
    return(person_color)


def draw(anno_info, vid_file):
    # Video capture
    vid_handler = cv2.VideoCapture(vid_file)

    # Process every frame
    frame_num = 0
    while vid_handler.isOpened():
        rc, frame_data = vid_handler.read()
        if rc == False:
            # End of video frames
            vid_handler.release()
        else:
            if frame_num in anno_info:
                for person_tuple in anno_info[frame_num]:
                    person_id, xmin, ymin, width, height = person_tuple
                    # Aspect ratio change
                    gt_width = 1920
                    gt_height = 1080
                    vid_height, vid_width, _ = frame_data.shape
                    xmin = int(xmin * (vid_width / gt_width))
                    ymin = int(ymin * (vid_height / gt_height))
                    width = int(width * (vid_width / gt_width))
                    height = int(height * (vid_height / gt_height))
                    xmax = xmin + width
                    ymax = ymin + height
                    # Draw bounding boxes
                    person_color = get_person_color(person_id)
                    cv2.rectangle(frame_data, (xmin, ymin), (xmax, ymax), person_color, 2)
                    cv2.imwrite("frame_{}.jpg".format(frame_num), frame_data)
            frame_num += 1


def main():

    # Base paths
    tcc_path = "/home/bruno/Downloads/tcc"
    track_path = os.path.join(tcc_path, "track")

    # Ground truth and video paths
    dataset = "MOT20-01"
    anno_path = os.path.join(track_path, "data/anno/{}/gt".format(dataset))
    anno_file = os.path.join(anno_path, "gt.txt")
    vid_file = os.path.join(track_path, "data/raw/{}.webm".format(dataset))

    # Parse ground truth
    anno_info = get_anno_info(anno_file)

    # Draw bounding boxes
    draw(anno_info, vid_file)


if __name__ == "__main__":
    main()
