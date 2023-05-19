from metaworld.task_config import TASK_DICK
import numpy as np
import random
import imageio
from pathlib import Path
from typing import List, Dict
import copy


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

        self.save_gif = save_gif
        if self.save_gif:
            self.img_list = []

    def reset_task_list(self, task_list : List[str]) -> None:
        """
        该接口暂时不使用
        重置任务列表/终止任务
        """
        assert isinstance(task_list, list)
        print(f"reset_task_list: {task_list}")
        self.env.reset_task_list(task_list)

    def append_task_list(self, task_list : List[str]) -> None:
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
            img = self.env.render(offscreen=True, camera_name=camera_name, resolution=(320, 240))
            img_dict[camera_name] = img
        return img_dict

    def _get_demo_img(self) -> np.ndarray:
        img = self.env.render(offscreen=True, camera_name='corner3')
        if self.save_gif:
            self.img_list.append(img)
        return img

    def env_step(self) -> np.ndarray:
        print(self.env.task_list)
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

    def gif_save(self, name : str = None) -> None:
        """
        当前 img_list 存成 gif 文件
        """
        if name is None:
            name = 'demo'
        if self.save_gif:
            print(f'saving gif at {self.step} ...')
            root_path = Path(__file__).parent / 'data'
            root_path.mkdir(exist_ok=True, parents=True)
            imageio.mimsave(str(root_path / (name + '.gif')), self.img_list, duration=40)
            self.img_list = []


if __name__ == "__main__":
    task_name = 'display'
    seed = 0
    demo = Demo(task_name=task_name, seed=seed, save_gif=True)
    # done = False
    # test_task_dict = {
    #     10: ['desk-pick', 'coffee-push', 'coffee-button', 'coffee-pull', 'desk-place'],
    #     1000: ['desk-pick', 'drawer-place'],
    #     2000: 'reset',
    #     2050: 'stop'
    # }
    test_task_dict = {
        10: ["desk-pick", "drawer-place"],
        500: 'reset',
        510: ["desk-pick", "coffee-push", "coffee-button", "coffee-pull", "desk-place"],
        650: 'stop'
    }
    # test_task_dict = { # test error
    #     10: ['desk-pick', 'coffee-push', 'coffee-pull', 'coffee-button'],
    #     1000: 'stop'
    # }
    step = 0
    while True:
        """
        目前暂不支持对书架进行操作
        支持的任务列表：[
            'coffee-button',
            'coffee-pull',
            'coffee-push',
            'drawer-close',
            'drawer-open',
            'drawer-pick',
            'drawer-place',
            'desk-pick',
            'desk-place',
            'reset'
            ]
        """
        img = demo.env_step()
        # TODO 推流
        # TODO 获取指令instruction
        # reset 表示重置环境 stop 表示停止演示
        instruction = test_task_dict.get(step, None)
        if isinstance(instruction, list):
            demo.append_task_list(instruction)
        elif isinstance(instruction, str):
            if instruction == 'reset':
                demo.reset()
            elif instruction == 'stop':
                break
            else:
                raise ValueError(f"Error instruction: {instruction}")
        step += 1
        if demo.over():
            raise Exception(f"Task {demo.now_task} Policy failed.")
    demo.gif_save()