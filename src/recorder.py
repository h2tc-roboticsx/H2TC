'''
recorder.py

Lin Li (linli.tree@outlook.com) / 21 Sep. 2022 (last modified)

The recorder program controls mutliple ZED cameras, Event camera, OptiTrack and Hand Engine to record.

Contributions:

The main procedures was written by Lin with the help from Jianing Qiu.
The remote display part was written by Xiaoma.

Code architecture:

The entire recording system includes 3 ZED cameras, 1 Event camera, 1 Hand Engine, 1 OptiTrack. 
This recorder master maintains 4 child processes for 3 ZED and 1 Event cameras recording respectively.
It controls the recording of these child processes by changing the value of an integer variable shared between the parent and child processes.
It controls (start and stop) the recording of Hand Engine and NatNet (OptiTrack client) via sending UDP message
Meanwhile, it maintains a thread for listening the message from the client i.e. optitrack client NatNet

'''

import os, sys, socket, signal, shutil, time
from warnings import warn
from threading import Thread
from subprocess import run, Popen, PIPE, STDOUT
import multiprocessing as mp
from argparse import ArgumentParser
from addict import Dict
import requests
import re

# get the absolute path of the current and root directory
curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")

# add root directory to the system path
sys.path.append(root_path)

from src.utils import zed
from src.utils import event as evt
from src.utils import optitrack as opi
from src.log import Log, SUBJECTS, OBJECTS, data_path, CHN


'''
Hyper-parameters and constant

'''

# convert the IP address and port string to the tuple
# e.g. 192.168.10.41:300 -> (192.168.10.41, 300)
def str_2_addr(addr):
    ip, port = addr.split(':')
    return (ip, int(port))

argparser = ArgumentParser(description="Recording trigger server")
argparser.add_argument("--addr", type=str_2_addr, default="10.41.206.138:3003",
                       help="ip address and port of the current machine for UDP")
argparser.add_argument("--he_addr", type=str_2_addr, default="10.41.206.141:30039",
                       help="ip address and port of the Hand Engine machine for UDP")
argparser.add_argument("-l", "--length", default=5,
                       help="the length of recording")
argparser.add_argument("--nposition", type=int, default=16,
                       help="num of position squares to locate subjects")
argparser.add_argument("--clients", nargs='+', choices=['optitrack'], default=['optitrack'],
                       help="clients allowed to communicate")
argparser.add_argument("-n", "--zed_num", type=int, default=3,
                       help="number of ZED cams for recording")
argparser.add_argument("--fps", type=int, default=60,
                       help="FPS for ZED recording")
argparser.add_argument("-r", "--resolution", type=str, default='720p',
                       help="resolution for ZED recording")
argparser.add_argument("-t", "--tolerance", type=float, default=0.1,
                       help="frame drop tolerance")

# buffer size of UDP message
BUFFER_SIZE = 100

DIVISION_LINE_LENG = 60 # length of the division line printed in the console

CMD = Dict({
    'register' : '2',
    'next' : '1',
    'repeat' : '0',
    'stop' : '-1',
    'confirm' : '-2'
})

CMD.HE = Dict({
    'start' : '<?xml version="1.0" encoding="UTF-8" standalone="no" ?><CaptureStart><Name VALUE="{}"/></CaptureStart>',
    'stop' : '<?xml version="1.0" encoding="utf-8"?><CaptureStop><Name VALUE="{}" /></CaptureStop>'
})

# clients' UDP addresses
# key: client ID
# value: (IP, port) tuple
ADDRS = Dict({})


def listen(socket, clients):
    '''
    listening to the UDP messages from the client, NatNet, and itself

    the message from NatNet is used to confirm the connection between the master recorder and it

    the message from this program itself is used to safely terminate this thread

    Returns:
        None

    Args:
        socket (socket.Socket): UDP socket
        clients (list): a list of clients to connect

    '''
    
    print("waiting for clients to connect")
    while True:
        # receive UDP message
        msg, addr = socket.recvfrom(BUFFER_SIZE)
        # decode message to string format
        msg = msg.decode()
        # break the loop to end the thread when stop msg received
        if msg == CMD.stop: break

        # receive msg i.e. client id from the clients
        # store the client id and addr if never stored before
        if msg in clients and msg not in ADDRS:
            ADDRS[msg] = addr
            print("{} connected from {}:{}".format(msg, *addr))
            # send confirmation msg to the client to build the connection
            socket.sendto(CMD.confirm.encode(), addr)
    print("stop listening")


def register():
    '''
    check if subjects are valid i.e. available in the register/subjects.csv

    Returns:
        list: two ID string of the subject 1 (wearing sensors) and 2

    '''
    
    sub1_invalid, sub2_invalid = True, True
    while sub1_invalid or sub2_invalid:
        print("type subjects (equipped first), separated by the space, to register:")
        # get input and separate by the space
        subjects = str(input()).split(' ')
        # invalid if the number of input(subjects) doesn't equal to 2
        if len(subjects) != 2: continue

        # check if subject in the pre-defined subject list
        sub1_invalid, sub2_invalid = [subject not in SUBJECTS for subject in subjects]
    return subjects



def record(take_id, rec_enableds, length, he_addr, opti_addr):
    '''
    Launch devices to record the data

    Returns:
        None
    
    Args:
        take_id (string): ID of the take
        rec_enableds (list): list of rec_state variable shared between the parent and child processes to control the reording
        length (int): length of recording
        opti_addr (tuple): IP address and port of NatNet client.

    '''
    
    # black the screen showing instruction for the helmet wearer
    requests.post('http://10.41.206.169:5000/command', '')
    requests.post('http://localhost:5000/command', '')

    # send start cmd via UDP to Hand Engine application to start recording
    udp_socket.sendto(bytes(CMD.HE.start.format(take_id), 'utf-8'), he_addr)
    # printing the current timestamp in the console
    print('{}:\t{}'.format("HE start", time.time_ns()))

    # send take id via UDP to the client of optitrack to start recording the stream
    udp_socket.sendto(take_id.encode(), opti_addr)
    print('{}:\t{}'.format("Optitrack start", time.time_ns()))

    # run the program 'spd-say' to speak the start signal to subjects
    Popen(['spd-say', 'kai shi'])

    # change the recording states to 1 to start recording
    for rec_enabled in rec_enableds: rec_enabled.value = 1
    # sleep the main process for the recording length time
    time.sleep(length)
    # change the recording states to 0 to stop recording
    for rec_enabled in rec_enableds: rec_enabled.value = 0
    
    print('{}:\t{}'.format("Depth stop/HE stop", time.time_ns()))
    # send stop cmd via UDP to Hand Engine
    udp_socket.sendto(bytes(CMD.HE.stop.format(take_id), 'utf-8'), args.he_addr)
    
    # run the program 'spd-say' to speak the stop signal to subjects
    run(['spd-say', 'jie shu'])

        
def end():
    global procs
    global rec_states
    global udp_socket
    # address for all clients plus server's
    addrs = list(ADDRS.values()) + [args.addr]
    # send stop cmd to end the clients and the listening thread in the server
    for addr in addrs: udp_socket.sendto(CMD.stop.encode(), addr)

    # close the thread listening to the clients
    udp_socket.close()

    # change the recording states to -1 to terminate the infinite loops in the children processes
    # so that they can finish
    for state, proc in zip(rec_states, procs):
        state.value = -1
        proc.join()
        proc.close()
        
    exit()

    
'''
Command Interface

'''
 
    
if __name__ == '__main__':
    global args
    global udp_socket
    global procs
    global rec_states
    args = argparser.parse_args()

    # init udp socket to send and receive commands
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.bind(args.addr)

    # listen to the UDP message from the clients in a separate thread
    thread = Thread(target=listen, args=(udp_socket, args.clients,))
    thread.start()

    # set up multiprocessing start method for launching the processes for ZED and Event cameras
    mp.set_start_method('spawn')

    # init an array of recording states for cameras shared between the parent and the child processes
    # the size of array = number of ZEDs + 1 (event)
    # 1: recording on
    # 0: recording off
    # -1: terminate the process
    rec_states = [mp.Value('i', 0) for _ in range(args.zed_num+1)]
    take_queue = [mp.Queue() for _ in range(args.zed_num+1)]
    # init Event child process with the function wai_record from evt module
    # and the parameters: (recording state, queue)
    procs = [mp.Process(target=evt.wait_record, args=(rec_states[0], take_queue[0]))]
    # init N ZED children processes with the function wait_record from zed module
    # and the parameters: (recording state, camera ID, queue, FPS, resolution)
    for cam_id, rec_state, take_q in zip(zed.CAM_IDS, rec_states[1:], take_queue[1:]):
        procs.append(mp.Process(target=zed.wait_record,
                                args=(rec_state, cam_id, take_q, args.fps, args.resolution)))
    # start all children processes to initilize all devices and wait for the recording start command
    for proc in procs: proc.start()
    
    
    # init logbook to log the instructions and the results.
    log = Log()
    # register subjects to be recorded
    subjects = register()

    # while-loop interface to continuously parse user's input and execute the command
    # exit as stop command received
    while True:
        # print a line to separate the output in the console
        print('-' * DIVISION_LINE_LENG)
        print("pressing {} to register subjects".format(CMD.register))
        print("pressing {} to start NEXT recording".format(CMD.next))
        print("pressing {} to REPEAT the last recording setting".format(CMD.repeat))
        print("pressing {} to abort".format(CMD.stop))
        
        # get input from user
        msg = str(input())

        # filter out invalid commands
        if msg not in CMD.values():
            warn("invalid command: {}".format(msg))
            continue

        # end the program
        if msg == CMD.stop: end()

        # register the new subjects
        if msg == CMD.register:
            subjects = register()
            continue

        # continue to the next iteration if any client fails to connect
        if any([client not in ADDRS for client in args.clients]):
            warn("Missing network address for the client")
            continue                

        # load next take if the cmd is Next
        if msg == CMD.next:
            take = log.next_take(subjects)
            # continue to the next iteration if no more take to record
            if take is None:
                print("all takes finished for subject: {}".format(subjects[0]))
                print("register new subject to continue.")
                continue

        # form here, record the current take

        # compose the directory path of the current take
        take_path = os.path.join(data_path, take.ID)
        
        # remove the existing take path if it exists
        # this could be the incomplete takes caused by interruption
        if os.path.isdir(take_path):
            print("Path to store the recording for this take already existed.")
            shutil.rmtree(take_path)

        # create the directory in the file system
        os.mkdir(take_path)

        print('-' * DIVISION_LINE_LENG)
        # display take ID and the object to be used
        print("Recording {}: {}".format(take.ID, take.obj))

        # sub1 corresponding the one wearing the sensors
        sub1, sub2 = subjects

        # init printings for two subjects' instructions
        # subject ID: position action
        # CHN is a dictionary mapping English to Chinese
        info = ["{0: <10}:\t{1} {2} ".format(sub, take[sub].position, CHN[take[sub].action])
                for sub in subjects]

        # add 'hand' (single or both) info to the subject1's printing if has
        if take[sub1].hand is not None: info[0] += CHN[take[sub1].hand]
        # add 'hand' and 'height' (vertical position of the hand) to subject2's
        info[1] += "{} {} ".format(CHN[take[sub2].hand], CHN[take[sub2].height])
        # add 'speed' info if subject2 throws else 'horizon' (horizontal position of the hand)
        info[1] += CHN[take[sub2].speed] if take[sub2].action=='throw' else CHN[take[sub2].horizon]
        # print the information
        for inf in info: print(inf)

        # TODO: written by Xiaoma 
        # display the instructions on the external displayers
        # by sending the post request
        sub1_screen_info = re.split('[(|)]', take[sub1].position)[1]
        if take[sub1].hand == '' or take[sub1].hand == None:
            sub1_screen_info += ', m'
        elif take[sub1].hand == 'single':
            sub1_screen_info += ', s'
        elif take[sub1].hand == 'both':
            sub1_screen_info += ', d'

        if take[sub1].action == 'throw':
            sub1_screen_info += ', p'
        elif take[sub1].action == 'catch':
            sub1_screen_info += ', j'

        requests.post('http://10.41.206.169:5000/command', sub1_screen_info)

        sub2_screen_info = re.split('[(|)]', take[sub2].position)[1]
        if take[sub2].hand == '' or take[sub2].hand == None:
            sub2_screen_info += ', m'
        elif take[sub2].hand == 'single':
            sub2_screen_info += ', s'
        elif take[sub2].hand == 'both':
            sub2_screen_info += ', d'

        if take[sub2].action == 'throw':
            sub2_screen_info += ', p'
            if take[sub2].speed == '' or take[sub2].speed == None:
                sub2_screen_info += ', -1'
            elif take[sub2].speed == 'fast':
                sub2_screen_info += ', 0'
            elif take[sub2].speed == 'slow':
                sub2_screen_info += ', 1'
            elif take[sub2].speed == 'normal':
                sub2_screen_info += ', 2'
            
        elif take[sub2].action == 'catch':
            sub2_screen_info += ', j'
            if take[sub2].horizon == '' or take[sub2].horizon == None:
                sub2_screen_info += ', -1'
            elif take[sub2].horizon == 'left':
                sub2_screen_info += ', 0'
            elif take[sub2].horizon == 'middle':
                sub2_screen_info += ', 1'
            elif take[sub2].horizon == 'right':
                sub2_screen_info += ', 2'

        if take[sub2].height == '' or take[sub2].height == None:
            sub2_screen_info += ', -1'
        elif take[sub2].height == 'overhead':
            sub2_screen_info += ', 0'
        elif take[sub2].height == 'overhand':
            sub2_screen_info += ', 1'
        elif take[sub2].height == 'chest':
            sub2_screen_info += ', 2'
        elif take[sub2].height == 'underhand':
            sub2_screen_info += ', 3'

        requests.post('http://localhost:5000/command', sub2_screen_info)
        
         
        # pause and wait for informing subjects the action setting
        print("press 0 to skip, any other key to continue")
        cmd = input()
        if cmd == '0': continue

        # put take path into the shared queue so that the children processes can access
        for q in take_queue: q.put(take_path)
        # call record() to start recording
        record(take.ID, rec_states, args.length, args.he_addr, ADDRS.optitrack)

        try:
            # sleep the main process to wait for the completion of all children processes
            time.sleep(1)
            # raise an exception if any recording data missed for ZED
            zed.verify_file_integrity(take_path)
            # raise an exception if any recording data missed for Event
            evt.verify_file_integrity(take_path)

            # verify OptiTrack data for every 5 recordings
            if int(take.ID) % 5 == 0:
                # raise an exception if frames drop over the threshold
                opi.verify_integrity(take_path, opi.DEFAULT_FPS*args.length, args.tolerance)
                # convert coordinates from global to local
                obj_ids = opi.convert(os.path.join(take_path, opi.DEFAULT_FILE_NAME), t_matrix_type='global')
                # visualize optitrack trajectory
                evo_plot = ['evo_traj', 'tum']
                for obj_id in obj_ids: evo_plot.append(os.path.join(take_path, str(obj_id)+'.txt'))
                evo_plot.extend(['-p', '--plot_mode', 'zy'])
                run(evo_plot)

        except Exception as e:
            # print the exception info if have
            # and continue to the next iteration to not save this take in the logbook
            print(e)
            continue
                    
        # post recording processing
        print("recording finished, type result (1: success, 0: failed, -1: invalid):")
        result = None
        # repeatedly ask for input as the result of throw-catch until a valid value
        # 1: success
        # 0: failed
        # -1: invalid
        while result not in ('0', '1', '-1'): result = str(input())

        # remove the take and continue to the next iteration
        # if the take is identified as invalid
        if result == '-1':
            print("recording data is REMOVED since identified as invalid")
            shutil.rmtree(take_path)
            continue
        
        # log the take information into the logbook
        for sub in subjects:
            log.log(sub, success=result, take_id=take.ID, **take[sub])
            
        # save the logbook
        log.finish(take.ID)
