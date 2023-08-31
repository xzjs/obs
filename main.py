import json
import os
import time

import requests
from Metaworld.demo import Demo
import numpy as np
import redis
import ffmpeg
from threading import Thread, current_thread
import logging
from multiprocessing import Process, Pipe

logging.basicConfig(filename='./main.log', level=logging.INFO)
CAMERA_LIST = ["corner3", "corner", "corner2", "topview"]
demo = None


def push(uid):
    '''
    推流函数
    '''
    task_name = 'display'
    seed = 0
    global demo
    demo = Demo(task_name=task_name, seed=seed)
    frame = demo.env_step()
    height, width, channels = frame.shape
    name = f"arm{uid}"
    url = f"rtmp://172.18.116.126/myapp/{name}"
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{width}x{height}")
        .output(url, f='flv', crf=18, preset='slower', vcodec='libx264', tune='zerolatency')
        .global_args("-re")
        .run_async(pipe_stdin=True)
        # .run_async(pipe_stdin=True, quiet=True)
    )
    r = requests.post('http://172.18.116.126/api/model/start/', json={
        "user": name,
        "path": f"http://172.18.116.126/live?app=myapp&stream={name}"
    })
    logging.info("response:"+r.text)
    logging.info("start push stream "+url)
    while True:
        try:
            frame = demo.env_step()
            process.stdin.write(frame.astype(np.uint8).tobytes())
            time.sleep(0.02)
        except:
            pass


def arm(uid):
    '''
    监听命令函数
    '''
    r = redis.Redis(host='localhost', port=6379,
                    decode_responses=True, password='123456')
    pub = r.pubsub()
    pub.subscribe('task_list')
    msg_stream = pub.listen()
    global demo
    thread = Thread(target=push, kwargs={'uid': uid}, daemon=True)
    thread.start()
    while (demo == None):
        pass
    for msg in msg_stream:
        if msg["type"] == "message":
            data = json.loads(msg["data"])
            if uid == data['uid']:
                action = data['task_list']
                if action == 'reset':
                    demo.reset()
                elif action == 'stop':
                    logging.info(f"{os.getpid()} stop,arm{uid} stop")
                    break
                else:
                    demo.append_task_list(action)


if __name__ == "__main__":
    processes = {}
    r = redis.Redis(host='localhost', port=6379,
                    decode_responses=True, password='123456')
    pub = r.pubsub()
    pub.subscribe('task_list')
    msg_stream = pub.listen()
    for msg in msg_stream:
        if msg["type"] == "message":
            logging.info(msg["data"])
            data = json.loads(msg["data"])
            uid = data['uid']
            # 启动进程
            if (uid not in processes) or (processes[uid].is_alive() == False):
                p = Process(
                    target=arm, kwargs={'uid': uid}, name=f"arm{uid}")
                p.daemon = True
                p.start()
                logging.info(f'arm{uid}进程启动')
                processes[uid] = p
        elif msg["type"] == "subscribe":
            logging.info('订阅成功')
