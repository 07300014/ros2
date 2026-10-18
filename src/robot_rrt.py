import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

# 节点类
class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

# RRT*主类
class RRTStar3D:
    def __init__(self, start, goal, obstacle_list, space_bounds, step_size=1.0, max_iter=600, search_radius=2.0):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.obstacle_list = obstacle_list
        self.space_min, self.space_max = space_bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.node_list = [self.start]

    # 随机采样
    def random_sample(self):
        x = random.uniform(self.space_min[0], self.space_max[0])
        y = random.uniform(self.space_min[1], self.space_max[1])
        z = random.uniform(self.space_min[2], self.space_max[2])
        return Node(x, y, z)

    # 找最近节点
    def nearest_node(self, rand_node):
        dist_list = [(node.x - rand_node.x)**2 + (node.y - rand_node.y)**2 + (node.z - rand_node.z)**2 for node in self.node_list]
        min_idx = dist_list.index(min(dist_list))
        return self.node_list[min_idx]

    # 生成新节点
    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        dist = math.hypot(dx, dy, dz)
        if dist < self.step_size:
            return to_node
        new_x = from_node.x + dx / dist * self.step_size
        new_y = from_node.y + dy / self.step_size
        new_z = from_node.z + dz / self.step_size
        return Node(new_x, new_y, new_z)

    # 碰撞检测
    def collision_check(self, node1, node2):
        for obs in self.obstacle_list:
            ox, oy, oz, r = obs
            # 线段到球障碍物距离检测
            dx_seg = node2.x - node1.x
            dy_seg = node2.y - node1.y
            dz_seg = node2.z - node1.z
            a = dx_seg**2 + dy_seg**2 + dz_seg**2
            if a == 0:
                continue
            t = max(0, min(1, ((ox-node1.x)*dx_seg + (oy-node1.y)*dy_seg + (oz-node1.z)*dz_seg)/a))
            nearest_x = node1.x + t * dx_seg
            nearest_y = node1.y + t * dy_seg
            nearest_z = node1.z + t * dz_seg
            dist_obs = math.sqrt((ox-nearest_x)**2 + (oy-nearest_y)**2 + (oz-nearest_z)**2)
            if dist_obs <= r:
                return True
        return False

    # 查找邻域节点
    def find_neighbors(self, new_node):
        neighbors = []
        for node in self.node_list:
            d = math.sqrt((node.x-new_node.x)**2 + (node.y-new_node.y)**2 + (node.z-new_node.z)**2)
            if d < self.search_radius:
                neighbors.append(node)
        return neighbors

    # 选代价最小父节点
    def choose_parent(self, new_node, neighbors):
        min_cost = float('inf')
        best_parent = None
        for n in neighbors:
            if not self.collision_check(n, new_node):
                cost = n.cost + math.sqrt((n.x-new_node.x)**2 + (n.y-new_node.y)**2 + (n.z-new_node.z)**2)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = n
        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = min_cost
        return new_node

    # 重布线
    def rewire(self, new_node, neighbors):
        for n in neighbors:
            if new_node.cost + math.sqrt((n.x-new_node.x)**2 + (n.y-new_node.y)**2 + (n.z-new_node.z)**2) < n.cost:
                if not self.collision_check(new_node, n):
                    n.parent = new_node
                    n.cost = new_node.cost + math.sqrt((n.x-new_node.x)**2 + (n.y-new_node.y)**2 + (n.z-new_node.z)**2)

    # 主规划函数
    def plan(self):
        for _ in range(self.max_iter):
            rand_node = self.random_sample()
            nearest = self.nearest_node(rand_node)
            new_node = self.steer(nearest, rand_node)
            if not self.collision_check(nearest, new_node):
                neighbors = self.find_neighbors(new_node)
                new_node = self.choose_parent(new_node, neighbors)
                self.node_list.append(new_node)
                self.rewire(new_node, neighbors)
                # 判断是否靠近终点
                if math.sqrt((new_node.x-self.goal.x)**2 + (new_node.y-self.goal.y)**2 + (new_node.z-self.goal.z)**2) < self.step_size:
                    if not self.collision_check(new_node, self.goal):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + math.sqrt((new_node.x-self.goal.x)**2 + (new_node.y-self.goal.y)**2 + (new_node.z-self.goal.z)**2)
                        self.node_list.append(self.goal)
                        print("✅ 找到可行路径！")
                        return self.extract_path()
        print("❌ 未找到路径，可增大迭代次数")
        return None

    # 回溯提取路径
    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]

    # ===== 3D可视化绘图函数 =====
    def plot_result(self, path):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        # 画所有扩展树
        for node in self.node_list:
            if node.parent:
                ax.plot([node.x, node.parent.x],
                        [node.y, node.parent.y],
                        [node.z, node.parent.z], c='lightgray', lw=0.4)
        # 画障碍物（球体）
        for obs in self.obstacle_list:
            ox, oy, oz, r = obs
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
            x_sphere = ox + r*np.cos(u)*np.sin(v)
            y_sphere = oy + r*np.sin(u)*np.sin(v)
            z_sphere = oz + r*np.cos(v)
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='darkred', alpha=0.6)
        # 画最终规划出的最优路径
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2], c='blue', lw=2.5, label='最优RRT*路径')
        # 起点终点标记
        ax.scatter(self.start.x, self.start.y, self.start.z, color='green', s=80, label='起点')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, color='orange', s=80, label='终点')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('RRT* 三维路径规划 移动机器人避障仿真')
        ax.legend()
        plt.savefig('rrt3d_result.png', dpi=300) # 自动保存图片，作业上传用
        plt.show()

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 起点、终点
    start_pos = [0, 0, 0]
    goal_pos = [9, 9, 5]
    # 障碍物列表 (x,y,z,半径)
    obs_list = [
        (3, 3, 2, 1.2),
        (6, 4, 3, 1.0),
        (4,7,2,1.3)
    ]
    # 空间范围 [min(x,y,z), max(x,y,z)]
    bounds = ([0,0,0], [10,10,6])
    planner = RRTStar3D(start_pos, goal_pos, obs_list, bounds, max_iter=800)
    path_result = planner.plan()
    if path_result is not None:
        planner.plot_result(path_result)
        print("路径坐标：")
        print(np.array(path_result))
