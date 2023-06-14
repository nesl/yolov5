import os
import cv2 
import re
import pdb
import math

#Extract images from videos and get the associated labels into separate frame.txt files

def get_ground_truths(ground_truth_file, previous_ground_truth, f_idx): #If bbox is in first frame, problems

    ground_truth_list = []
    next_ground_truth = []


    if previous_ground_truth:
        if int(previous_ground_truth[2]) == f_idx:
            ground_truth_list.append(previous_ground_truth)

            while True:
                
                next_ground_truth_tmp = ground_truth_file.readline()
                
                if len(next_ground_truth_tmp) == 0:
                    next_ground_truth = []
                    break
                    
                next_ground_truth = next_ground_truth_tmp.strip().split(',')
            
                if int(next_ground_truth[2]) == f_idx:
                    ground_truth_list.append(next_ground_truth)
                else:
                    break
        else:
            next_ground_truth = previous_ground_truth
        
    else:
        next_ground_truth = ground_truth_file.readline().strip().split(',')
                
    return ground_truth_list, next_ground_truth
    
def get_camera_idx(cameras, k):

    camera_idx = -1
    for c_idx,c in enumerate(cameras):
        if c[0] == int(k):
            camera_idx = c_idx
            
    return camera_idx

    
frame_path = "frames/"
if not os.path.exists(frame_path):
    os.makedirs(frame_path)
    
labels_path = "labels/"
if not os.path.exists(labels_path):
    os.makedirs(labels_path)

root_directory = "/media/nesl/Elements/data/"

cam_num_file = re.compile(r'_cam(\d+)_')
cam_num_csv = re.compile(r'NAI (\d+)')


for dir_name in os.listdir(root_directory):
    take_dir = root_directory+dir_name+'/'
    if any("smoke" in take_dir+vid for vid in os.listdir(take_dir)):
        previous_ground_truth = []
        ground_truth_list = []
        cameras = []
        ground_truth_file = ""
        for vid in os.listdir(take_dir):
            vid_name = take_dir+vid
            if '.mp4' in vid and "smoke" in vid:
                
                mo = cam_num_file.search(vid)
                
                cap = cv2.VideoCapture(vid_name)
                pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                camera = [int(mo.group(1)), cap, pixel_width, pixel_height, fps]
                cameras.append(camera)
            elif '.csv' in vid:
                ground_truth_file = open(vid_name)
                ground_truth_file.readline()
                previous_ground_truth = ground_truth_file.readline().strip().split(',')
        
        if ground_truth_file:
        
            ground_truth_list, previous_ground_truth = get_ground_truths(ground_truth_file,previous_ground_truth,int(previous_ground_truth[2]))
            
            while True:

                g_bboxes = {}
                for g_bbox in ground_truth_list:
                    mo = cam_num_csv.search(g_bbox[3])
                    if mo:
                        key = mo.group(1)
                        if key not in g_bboxes:
                            g_bboxes[key] = ""
                        camera_idx = get_camera_idx(cameras,key)
                        #g_bboxes[key] += "%d %f %f %f %f\n" % (int(g_bbox[6]),float(g_bbox[7].replace('(','')),(pixel_height-float(g_bbox[8].replace(')',''))),abs(float(g_bbox[9].replace('(',''))-float(g_bbox[11].replace('(',''))),abs(float(g_bbox[12].replace(')',''))-float(g_bbox[10].replace(')',''))))
                        g_bboxes[key] += "%d %f %f %f %f\n" % (int(g_bbox[6]),float(g_bbox[7].replace('(',''))/cameras[camera_idx][2],(pixel_height-float(g_bbox[8].replace(')','')))/cameras[camera_idx][3],abs(float(g_bbox[9].replace('(',''))-float(g_bbox[11].replace('(','')))/cameras[camera_idx][2],abs(float(g_bbox[12].replace(')',''))-float(g_bbox[10].replace(')','')))/cameras[camera_idx][3])


                for k in g_bboxes.keys():
                    
                    camera_idx = get_camera_idx(cameras,k)
                    frame_id = math.trunc(float(g_bbox[1])*cameras[camera_idx][4])
                    print(k,frame_id, dir_name, g_bboxes[k])
                    
                    cameras[camera_idx][1].set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, image = cameras[camera_idx][1].read()
                    if ret:
                        cv2.imwrite(frame_path+dir_name[5:]+'_'+ str(cameras[camera_idx][0]) +'_'+str(frame_id) + '.jpg', image)

                        label_file = open(labels_path+dir_name[5:]+'_'+ str(cameras[camera_idx][0]) +'_'+str(frame_id) + '.txt', 'w')
                        label_file.write(g_bboxes[k])
                        label_file.close()
                        
                        #cv2.imshow("Frame", image)
                        #cv2.waitKey(1)

                if not previous_ground_truth:
                    break
                    
                ground_truth_list, previous_ground_truth = get_ground_truths(ground_truth_file,previous_ground_truth,int(previous_ground_truth[2]))
                

                
               
        
        



