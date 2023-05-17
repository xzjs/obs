import json
import time

import requests
from metaworld.task_config import TASK_DICK
import numpy as np
import random
import imageio
from pathlib import Path
from typing import List, Dict
import redis
import ffmpeg
import copy
from threading import Thread, current_thread


CAMERA_LIST = ["corner3", "corner", "corner2", "topview"]


class Demo(object):
    def __init__(self, task_name, seed=None, save_gif=False) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.env = TASK_DICK[task_name]['env'](seed=seed)
        self.policy = TASK_DICK[task_name]['policy']()
        self.obs = self.env.reset()
        self.info = {}
        self.done = False
        self.obs_img = None
        self.step = 0
        self.task_step = 0
        self.task_max_step = 500
        self.now_task = None
        self.task_list = []
        self.process = None
        self.stop = True

        self.save_gif = save_gif
        if self.save_gif:
            self.img_list = []

    def reset_task_list(self, task_list: List[str]) -> None:
        """
        重置任务列表/终止任务
        """
        self.task_list = task_list
        # self.env.reset_task_list(task_list)
        if self.process:
            self.process.terminate()  # 停止子进程

    def append_task_list(self, task_list: List[str]) -> None:
        """
        在当前任务列表后添加任务列表
        """
        assert isinstance(task_list, list)
        print(f"append_task_list: {task_list}")
        self.env.append_task_list(task_list)

    def reset(self) -> None:
        """
        重置环境
        """
        self.obs = self.env.reset()
        self.info = {}
        self.done = False
        self.obs_img = None
        self.step = 0
        self.task_step = 0
        self.now_task = None
        # 重置策略 / 考虑模型是否有重置操作
        self.policy.reset()

    def _get_obs_img(self) -> Dict[str, np.ndarray]:
        img_dict = {}
        for camera_name in ["corner3", "corner", "topview"]:
            img = self.env.render(
                offscreen=True, camera_name=camera_name, resolution=(320, 240))
            img_dict[camera_name] = img
        return img_dict

    def _get_demo_img(self) -> np.ndarray:
        img = self.env.render(offscreen=True, camera_name='corner3')
        if self.save_gif:
            self.img_list.append(img)
        return img

    def env_step(self) -> np.ndarray:
        # print(self.env.task_list)
        self.obs_img = self._get_obs_img()
        # TODO 模型前向代替policy
        action = self.policy.get_action(self.obs, self.now_task, self.info)
        action = np.clip(action, -1, 1)

        last_task = self.info.get('task_name', None)
        self.obs, reward, self.done, self.info = self.env.step(action)
        self.now_task = self.info['task_name']
        self.step += 1
        if last_task != self.now_task:
            self.task_step = 0
        else:
            self.task_step += 1
        return self._get_demo_img()

    def over(self) -> bool:
        """
        判断任务失败
        """
        if self.info.get('task_name', None) is None:
            return False
        return self.task_step > self.task_max_step

    def gif_save(self, name: str = None) -> None:
        """
        当前 img_list 存成 gif 文件
        """
        if name is None:
            name = 'demo'
        if self.save_gif:
            print(f'saving gif at {self.step} ...')
            root_path = Path(__file__).parent / 'data'
            root_path.mkdir(exist_ok=True, parents=True)
            imageio.mimsave(str(root_path / (name + '.gif')),
                            self.img_list, duration=40)
            self.img_list = []

    def push(self) -> None:
        frame = self.env_step()
        height, width, channels = frame.shape
        key = int(time.time())
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(f'rtmp://172.18.116.126/myapp/test666', f='flv', crf=19, preset='slow', vcodec='libx264')
            .global_args("-re")
            .run_async(pipe_stdin=True)
        )
        self.process = process

        r = requests.post('http://172.18.116.126/api/model/start/', json={
                          "user": "alex",
                          "path": "http://172.18.116.126/live?app=myapp&stream=test666"
                          })
        print('*'*5, "start push stream*****************", r.status_code, '*'*5)

        try:
            while not self.stop:
                frame = self.env_step()
                process.stdin.write(frame.astype(np.uint8).tobytes())
                # time.sleep(0.04)
        except Exception as e:
            print(e)
            self.stop = True
        process.stdin.close()
        process.wait()
        print('*'*5, 'thread stop', self.stop, '*'*5)


if __name__ == "__main__":
    task_name = 'display'
    seed = 0
    demo = Demo(task_name=task_name, seed=seed)
    task_list = []
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    pub = r.pubsub()
    pub.subscribe('task_list')
    msg_stream = pub.listen()
    for msg in msg_stream:
        if msg["type"] == "message":
            print(msg["data"])
            try:
                if msg['data'] == 'reset':
                    demo.reset()
                elif msg['data'] == 'stop':
                    demo.stop = True
                else:
                    task_list = json.loads(msg["data"])
                    demo.append_task_list(task_list)
                    if demo.stop:
                        demo.stop = False
                        thread1 = Thread(target=demo.push)
                        thread1.start()
            except Exception as e:
                print(e)
        elif msg["type"] == "subscribe":
            print(msg["channel"], '订阅成功')
