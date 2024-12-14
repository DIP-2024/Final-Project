import os, cv2, subprocess

vid_path = "./demo_src.mp4"

vid = cv2.VideoCapture(vid_path) 
if not os.path.exists('video_data'): 
    os.makedirs('video_data') 

cur = 0
while(1): 
    ret, frame = vid.read() 
    if ret: 
        name = './video_data/{}_'.format(vid_path[:-4]) + str(cur) + '.jpg'
        cv2.imwrite(name, frame) 
        cur += 1
    else: 
        break
 
vid.release() 
cv2.destroyAllWindows()

for i in range(1,len(os.listdir('./video_data'))):
    subprocess.run(["python", "detect.py", "--source", './video_data/demo_src_{}.jpg'.format(i)])

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
w, h = cv2.imread('./make_video/demo_src_0.jpg').shape[:2]
res = cv2.VideoWriter('demo.mp4', fourcc, 25, (h,w))

for i in range(len(os.listdir('./make_video'))):
    res.write(cv2.imread('./make_video/demo_src_{}.jpg'.format(i)))
cv2.destroyAllWindows()
res.release() 