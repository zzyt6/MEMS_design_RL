import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class grid_env():
    def __init__(self):
        self.rows = 15
        self.cols = 30
        self.first_position = np.array([int((self.rows-1)/2) + 1,0])
        self.end_position = np.array([int(self.rows-1),self.cols -1]) 
        self.current_position = self.first_position.copy()
        self.take_action = {#从0到3开始就是上左下右
            0:np.array([-1,0]),
            1:np.array([0,1]),
            2:np.array([1,0]),
            3:np.array([0,-1]),
        }
        self.state_end = False
        self.grid_env =np.zeros((self.rows, self.cols), dtype=int) 

    def action_space(self):

        return len(self.take_action)

    def state_space(self):
        return len(self.end_position)

    def __get_obs(self):
        return self.current_position.copy()

    def __get_info(self):
        return { 
            "distance":np.linalg.norm(self.current_position - self.end_position)
        }

    def reset(self):
        self.state_end = False
        self.current_position = self.first_position.copy()#只复制当前的位置
        self.grid_env = np.zeros((self.rows, self.cols), dtype=int)
        self.grid_env[self.current_position[0],self.current_position[1]] = 1
        return self.current_position.copy(),self.__get_info()
    
    def step(self,action:int):
        prev_distance = np.linalg.norm(self.current_position - self.end_position)

        reward = -1

        self.current_position += self.take_action[action]
        #限定agent运动的范围
        if self.current_position[0] > self.rows-1 or self.current_position[0] <0 or self.current_position[1] > self.cols-1 or self.current_position[1] <0:
            self.current_position -= self.take_action[action]
            reward = -5
            new_distance = prev_distance
        else:
            new_distance = np.linalg.norm(self.current_position - self.end_position)
        
        # reward += 2*(prev_distance - new_distance)
        
        self.grid_env[self.current_position[0],self.current_position[1]] = 1


        if np.array_equal(self.current_position, self.end_position):
            self.state_end =True
            reward += 200
        else:
            self.state_end =False

        truncated = False
        return self.__get_obs(),reward,self.state_end,truncated,self.__get_info()

    def render(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(self.cols / 3, self.rows / 3),
                num="Grid Env",
                frameon=False,
            )
            # 去掉 figure 四周空白
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.set_position([0, 0, 1, 1])
            fig.patch.set_visible(False)  # Figure 背景也隐藏

        ax.clear()
        ax.set_facecolor("white")

        # 只保留黑色小网格
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)  # 上下颠倒以保持行号向下
        ax.grid(False)  # 关闭主网格
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.8)

        # 去掉刻度和外框
        ax.tick_params(which="both", left=False, bottom=False,
                    labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        visit_map = self.grid_env.copy()
        visit_map[tuple(self.end_position)] = 3
        visit_map[tuple(self.current_position)] = 2

        cmap = colors.ListedColormap([
            (0.95, 0.95, 0.95),
            (1.00, 0.90, 0.40),
            (0.85, 0.20, 0.20),
            (0.40, 0.80, 0.40),
        ])
        norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        ax.imshow(
            visit_map,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="none",
            extent=(-0.5, self.cols - 0.5, self.rows - 0.5, -0.5),
            animated=True,
        )

        return ax