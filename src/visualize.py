'''
annotate.py

Lin Li (linli.tree@outlook.com) / 21 Sep. 2022 (last modified)

Automation tool for annotating the processed data

Contributions:

The entire code was written by Lin. 


'''

import time
import os, sys, csv, json
import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import font
from math import sqrt
import datetime as dt

from PIL import Image, ImageTk, ImageDraw, ImageFont
from addict import Dict
from argparse import ArgumentParser

import multiprocessing as mp
from threading import Thread, Event

curr_path = os.path.dirname(os.path.abspath(__file__)) # the directory of the current file
cwp =  os.path.join(curr_path, "..")
# add root path to the system PATH to allow import other user-defined modules
sys.path.append(cwp)

from src.log import  *


'''
Arguments and Constants

'''

argparser = ArgumentParser()
argparser.add_argument('--take', type=int, default=None,
                       help="ID of take to be annotated. Set None to annotate the first take in the 'data' directory. This can be given with a single integer number for one take.")
argparser.add_argument('--datapath', type=str, default="Sample_Cases/data",
                       help='data path to all takes')
argparser.add_argument('--speed', type=int, default=120,
                       help='FPS for playing streams')


# display configuration of each stream
# key: stream ID
# values:
#     img_path: path template to the image
#     dev: device ID
#     ts_path: path to the timestamp file
#     position: position in the grid layout
STREAMS = Dict({
    'rgbd0_rgb' : {'img_path' : 'rgbd0/left_{:04d}.png',
                   'dev' : 'rgbd0',
                   'ts_path' : 'rgbd0_ts.csv',
                   'position' : (0, 0)},
    'rgbd0_depth' : {'img_path' : 'rgbd0/depth_{:04d}.png',
                     'dev' : 'rgbd0',
                     'ts_path' : 'rgbd0_ts.csv',
                     'position' : (1, 0)},
    'rgbd1_rgb' : {'img_path' : 'rgbd1/left_{:04d}.png',
                   'dev' : 'rgbd1',
                   'ts_path' : 'rgbd1_ts.csv',
                   'position' : (0, 2)},
    'rgbd1_depth' : {'img_path' : 'rgbd1/depth_{:04d}.png',
                     'dev' : 'rgbd1',
                     'ts_path' : 'rgbd1_ts.csv',
                     'position' : (1, 2)},
    'rgbd2_rgb' : {'img_path' : 'rgbd2/left_{:04d}.png',
                   'dev' : 'rgbd2',
                   'ts_path' : 'rgbd2_ts.csv',
                   'position' : (0, 1)},
    'rgbd2_depth' : {'img_path' : 'rgbd2/depth_{:04d}.png',
                     'dev' : 'rgbd2',
                     'ts_path' : 'rgbd2_ts.csv',
                     'position' : (1, 1)},
    'event' : { 'img_path' : 'event/{:04d}.jpg',
                'dev' : 'event',
                'ts_path' : 'event_frames_ts.csv',
                'position' : (0, 3)},
    'hand' : {'img_path' : 'hand_motion/{}.jpg',
              'dev' : 'left_hand_pose',
              'ts_path' : None,
              'position': (1, 3)}
})

data = {}


'''
Auxiliary functions

'''

def load_timestamp(path):
    '''
    Load timestamp file

    Returns:
        list: of int timestamps

    Args:
        path (string): path to the timestamp file

    '''
    
    with open(path, 'r') as f:
        # ignore the first row since it is the header
        return [int(ts) for ts in f.readlines()[1:]]
    
def init_data(proc_path):
    '''
    Initialize annotation data

    Returns:
        dict: alignment timestamps
        int: frame index
        dict: optitarck data

    Args:
        proc_path (string): path to the processed directory

    '''

    tss = {} # timestamps collection for all streams from the timestamp files
    for stream, cfg in STREAMS.items():
        dev = cfg.dev # device ID
        # skip the existing device and hand
        if dev in tss or stream=='hand': continue
        # load timestamp
        ts = load_timestamp(os.path.join(proc_path, cfg.ts_path))
        tss[dev] = ts

    frame = 0 # initial frame index

    align_path = os.path.join(proc_path, 'alignment.json') # path to the alignment file
    with open(align_path, 'r') as f: align = json.loads(f.read()) # load alignment data
    # aligned timestamps for each stream
    # key: device ID
    # value: dict of frames
    #     key: frame index
    #     value: dict
    #         key: frame, timestamp
    aligned = {} 
    for fnum, dev_ts in align.items(): # iterate every frame in alignment data
        for dev, ts in dev_ts.items(): # iterate every device in that frame
            # initialize device frame data if device not exists in aligned
            if dev not in aligned: aligned[dev] = [Dict() for _ in align.keys()]
            fnum = int(fnum) # string frame number to int

            # get frame index in timestamp file if device has the timestamp file
            # otherwise, use the frame index from alignment data
            aligned[dev][fnum].frame = tss[dev].index(ts) if dev in tss else fnum
            aligned[dev][fnum].timestamp = ts

    # optitack data
    # key: object ID
    # value: dict:
    #     key: timestamp
    #     value: optitrack data list
    opti = {}
    # only retrieve the optitrack data for the subjects' head motion
    # to get the location of two subjects and compute the flying speed of the thrown-away object
    for oid in ['sub1_head_motion', 'sub2_head_motion']:
        opti_path = os.path.join(proc_path, oid+'.csv') # path to optitrack data file
        with open(opti_path, 'r') as f:
            opti[oid] = {} # new sub dict for a object ID
            for line in f.readlines()[1:]: # iterate every row from 2nd
                line = line.split(',') # a list of separated data
                # timestamp as key, the rest data as value
                opti[oid][int(line[0])] = line[1:] 
                
    return aligned, frame, opti



'''
Interface UI

'''

def load_images(img_q, frames, tss, img_path_temp, nframes, font, size, stop_flag):
    for frame in frames:
        if stop_flag.is_set(): break
        
        # the frame index in the timestamp file as image ID
        img_id = tss[frame]['frame']
        ts = tss[frame]['timestamp'] # the timestamp of the frame
        img_path = img_path_temp.format(img_id) # path to the frame image
        img = Image.open(img_path) # load image
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        draw = ImageDraw.Draw(img) # drawing object
        if ts is None: # no timestamp available
            ts = ''
        else:
            # human-readable format of the timestamp
            date = dt.datetime.fromtimestamp(int(ts/1e9))
            date_str = date.strftime('%m.%d %H:%M:%S')
            ts = date_str + ':{:03d}'.format(int(ts%1e9/1e6))

        info = '{}/{} {}'.format(frame, nframes, ts) # timestamp information to display
        # draw the timestamp infomation on the image
        draw.text((10, 10), info, 'red', font=font)

        if not stop_flag.is_set():
            img_q.put((img_path_temp, frame, img))
        
def get_images(dev, images, img_q, img_path, nframes, stop_flag):
    while not stop_flag.is_set():
        try:
            _img_path, frame, img = img_q.get_nowait()
            if img_path != _img_path: continue
            images[frame] = img
            
            if frame == nframes - 1: break
        except:
            pass
        
class ImageBlock(tk.Frame):
    '''
    Frame image displayer

    '''

    # default system font in Ubuntu system
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 40)
    
    def __init__(self, master, dev, img_path, frame, tss, width=200, height=200):
        '''
        Initialization of the displayer

        Returns:
            None

        Args:
            master (tkinter): the parent tkinter container of the current one
            dev (string): device ID
            img_path (string): path tempalte to the frame images
            frame (int): current frame index
            tss (list): sequence of timestamps of the device 'dev'
            width (int): initial width of the displayer
            height (int): initial height of the displayer

        '''

        # initialize with the parent class's initializer
        super(ImageBlock, self).__init__(master,
                                         relief=tk.RAISED, # set the border with raised animation
                                         borderwidth=1,
                                         width=width,
                                         height=height)
        # the backbone displayer is a tk.Label
        self.display = tk.Label(master=self,
                                bg='white', # background color
                                width=width,
                                height=height)
        # set layout to be automatically adjusted according to the parent container
        self.display.pack(fill=tk.BOTH, expand=True)

        self.img_path = img_path
        self.frame = frame
        self.nframes = len(tss) # total number of frames
        self.tss = tss
        self.dev = dev
        
        # placehoder image collections
        self.imgs = [None for _ in range(self.nframes)]
        # placehoder photo (image) collections
        self.photos = [None for _ in range(self.nframes)]
        
        # bind <Configure> event (parent resize) to self.resize to customize the resize effect
        self.bind("<Configure>", self.resize)

        self.proc_stop = mp.Event()
        self.thread_stop = Event()
        self.img_q = None
                
    def resize(self, event):
        '''
        Resize the current UI. This is called when the parent UI changes its size.

        Returns:
            None

        Args:
            event (tkinter.Event): event of size change from the parent container
        
        '''

        # the new size to be resized to
        self.width, self.height = event.width, event.height
        
        self.img_q = mp.Queue()        
        self.proc = mp.Process(target=load_images,
                               args=(self.img_q,
                                     range(self.nframes),
                                     self.tss,
                                     self.img_path,
                                     self.nframes,
                                     self.FONT,
                                     (self.width-2, self.height-2),
                                     self.proc_stop))
        self.proc.start()

        self.t = Thread(target=get_images,
                        args=(self.dev,
                              self.imgs,
                              self.img_q,
                              self.img_path,
                              self.nframes,
                              self.thread_stop))
        self.t.start()
        
        self.display_frame(frame)
        
    def display_frame(self, frame):
        '''
        Display the image of a frame
        
        Returns:
            None
       
        Args:
            frame (int): the frame index whose image to display

        '''
        
        photo = self.load_photo(frame)
        # display the image
        self.display.configure(image=photo)
        self.display.image = photo
        
    def load_photo(self, frame):
        # the size of the current window
        size = (self.display.winfo_width(), self.display.winfo_height())
        # the photo of the frame
        photo = self.photos[frame]
        
        if photo is None:
            img = self.load_image(frame) # the frame image to display                
            photo = ImageTk.PhotoImage(img)
            # store photo for reuse
            self.photos[frame] = photo            

        return photo
        
    def load_image(self, frame):
        '''
        Load the image of a frame

        Returns:
            None

        Args:
            frame (int): the frame index

        '''
        
        # reuse the existing image if have
        if self.imgs[frame] is not None: return self.imgs[frame]
        
        while self.imgs[frame] is None: pass
    
        return self.imgs[frame]

    
    def forward(self):
        '''
        Forward a frame

        Returns:
            None

        Args:
            None

        '''

        # increase the frame index by 1
        self.frame = min(self.nframes-1, self.frame + 1)
        # display the new frame
        self.display_frame(self.frame)
        
    def backward(self):
        '''
        Backward a frame

        Returns:
            None

        Args:
            None

        '''

        # decrease the frame index by 1
        self.frame = max(0, self.frame - 1)
        # display the new frame
        self.display_frame(self.frame)
        
'''
Main program

'''


def autoplay(frame, nframes, displayers, speed, stop_flag):
    while not stop_flag.is_set() and frame < nframes:
        time.sleep(1/float(speed))
        for dis in displayers: dis.forward()
        frame += 1
        
if __name__ == '__main__':
    win = tk.Tk() # root window
    # 2 rows x 4 columns layout
    win.columnconfigure([0, 1, 2, 3], weight=1, minsize=200)
    win.rowconfigure([0, 1], weight=1, minsize=200)
    
    # parse arguments from console
    args = argparser.parse_args()
    
    # lx
    # folders
    data_path = args.datapath
    root_path = os.path.join(data_path, "..") # code depository root directory
    
    takes = [] # takes available in the local machine
    # an annotatable take should have at least all these data (folders)
    frame_dirs = ['event', 'rgbd0', 'rgbd1', 'rgbd2','hand_motion']
    for take in os.listdir(data_path): # iterate every take folder under data_path
        take_path = os.path.join(data_path, take)
        if all([os.path.exists(os.path.join(take_path, 'processed/'+f)) for f in frame_dirs]):
            # add the take id to the list if all specified data folders exit
            takes.append(take)

    if len(takes) == 0:
        exit(f'No valid takes in {data_path}')
    
    if args.take is None:
        args.take = takes[0]
    else:
        args.take = take_id_to_str(args.take)
            
    if args.take not in takes:
        exit(f'the required take (ID: {args.take}) is not available in {data_path}')
            
    take_id = args.take
    take_path = os.path.join(data_path, take_id)
    proc_path = os.path.join(take_path, 'processed') # the processed directory
    
    # get alignment, initial frame no., and optitrack data
    aligned, frame, opti = init_data(proc_path)
    data[take_id] = [aligned, frame, opti]
    
    # take ID as the title of the interface
    win.title(take_id)

    width = win.winfo_screenwidth() // 4 - 5
    height = win.winfo_screenheight() // 2 - 10
    
    # collection of displayers for each stream
    # key: stream ID
    # value: displayer UI
    displayers = {}
    for stream, cfg in STREAMS.items():
        # generate full image path template
        img_path = os.path.join(proc_path, cfg.img_path)

        # initialize displayer UI
        display = ImageBlock(win,
                             cfg.dev,
                             img_path,
                             frame,
                             aligned[cfg.dev],
                             width=width,
                             height=height)
        row, column = cfg.position
        # put displayer at the specified position in the window
        display.grid(row=row, column=column, padx=1, pady=1, sticky='nsew')
        displayers[stream] = display

    autoplay_stop_flag = None

    def on_closing():
        global autoplay_stop_flag, displayers
        if autoplay_stop_flag is not None:
            autoplay_stop_flag.set()

        for name, displayer in displayers.items():            
            displayer.proc_stop.set()
            displayer.thread_stop.set()
            
        for _, dis in displayers.items():
            while True:
                try:
                    dis.img_q.get(True, 0.1)
                except:
                    break
               
        win.destroy()
            
    def switch_frame(event):
        '''
        Forward or backward a frame in all displayers

        Key mapping
            'right-arrow': forward a frame
            'left-arrow': backward a frame

        Returns:
            None

        Args:
            event (tkinter.Event): key pressing event

        '''
        
        global displayers, autoplay_stop_flag

        if autoplay_stop_flag is not None:
            autoplay_stop_flag.set()
            autoplay_stop_flag = None
        
        # iterate every displayer
        for dev, dis in displayers.items(): 
            if event.keysym == 'Left':
                # backward a frame if <left-arrow> pressed
                dis.backward()
            else:
                # forward a frame if <right-arrow) pressed
                dis.forward()
                
        # update the current frame index by any displayer's current frame number
        data[take_id][2] = displayers[dev].frame

    def play_pause(event):
        '''
        Forward or backward a frame in all displayers

        Key mapping
            'right-arrow': forward a frame
            'left-arrow': backward a frame

        Returns:
            None

        Args:
            event (tkinter.Event): key pressing event

        '''
        
        global displayers, autoplay_stop_flag

        _displayers = list(displayers.values())
        displayer = _displayers[0]
        nframes = displayer.nframes

        if autoplay_stop_flag is None:
            autoplay_stop_flag = Event()
            t = Thread(target=autoplay,
                       args=(displayer.frame,
                             nframes,
                             _displayers,
                             args.speed,
                             autoplay_stop_flag))
            t.start()
        else:
            autoplay_stop_flag.set()
            autoplay_stop_flag = None
        
    # binding key pressing to the function
    win.bind("<Left>", switch_frame)
    win.bind("<Right>", switch_frame)
    win.bind("<space>", play_pause)

    win.protocol("WM_DELETE_WINDOW", on_closing)
    
    # launch the window
    win.mainloop()

