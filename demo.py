import numpy


class Demo:
    def __init__(self) -> None:
        '''
        构造函数
        '''
        pass

    def env_step(self) -> numpy.array:
        '''
        返回执行一步的图像
        '''
        pass

    def append_task_list(self, action) -> None:
        '''
        接收action命令
        '''
        pass
