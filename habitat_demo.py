import json
import logging
from multiprocessing import Process
import os
from threading import Thread
import time
import ffmpeg
import numpy as np
import redis
import requests
import habitat
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from demo import HabitatTask, HabitatDemo

logging.basicConfig(filename='./main.log', level=logging.INFO)
# logging.basicConfig(level=logging.INFO)
_demo = None


def push(uid):
    '''
    推流函数
    '''
    global _demo
    initial_task = HabitatTask("Nav", "Desk")
    with habitat.Env(config=initial_task.config.habitat) as env:
        _demo = HabitatDemo(env=env, env_spec=initial_task.env_spec)
        _demo.start()
        frame = _demo.env_step()
        height, width, channels = frame.shape
        name = f"habitat{uid}"
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
        while _demo.not_end:
            try:
                frame = _demo.env_step()
                process.stdin.write(frame.astype(np.uint8).tobytes())
                time.sleep(0.02)
            except:
                pass
        print('it is over')


def arm(uid):
    '''
    监听命令函数
    '''
    r = redis.Redis(host='localhost', port=6379,
                    decode_responses=True, password='123456')
    pub = r.pubsub()
    pub.subscribe('habitat')
    msg_stream = pub.listen()
    global _demo
    thread = Thread(target=push, kwargs={'uid': uid}, daemon=True)
    thread.start()
    while (_demo == None):
        pass
    for msg in msg_stream:
        if msg["type"] == "message":
            data = json.loads(msg["data"])
            if uid == data['uid']:
                action = data['task_list']
                if action == 'reset':
                    pass
                    # demo.reset()
                elif action == 'stop':
                    logging.info(f"{os.getpid()} stop,habitat{uid} stop")
                    break
                else:
                    for task in action:
                        if len(task) == 2:
                            _demo.append_task_list(
                                [HabitatTask(task[0], task[1])])


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    processes = {}
    r = redis.Redis(host='localhost', port=6379,
                    decode_responses=True, password='123456')
    pub = r.pubsub()
    pub.subscribe('habitat')
    msg_stream = pub.listen()
    for msg in msg_stream:
        if msg["type"] == "message":
            logging.info('redis-main:'+msg["data"])
            data = json.loads(msg["data"])
            uid = data['uid']
            # 启动进程
            if (uid not in processes) or (processes[uid].is_alive() == False):
                p = Process(
                    target=arm, kwargs={'uid': uid}, name=f"habitat{uid}")
                p.daemon = True
                p.start()
                logging.info(f'habitat{uid}进程启动')
                processes[uid] = p
        elif msg["type"] == "subscribe":
            logging.info('订阅成功')
