from mpl_toolkits import mplot3d
from matplotlib import cm


import numpy as np
import math
import matplotlib.pyplot as plt
import random


class PointXYZ():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __sub__(self, o):
        return np.array([self.x-o.x, self.y-o.y, self.z-o.z])
    def __str__(self):
        return str(self.x) + ',' + str(self.y) + ',' + str(self.z)
    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    def __mul__(self, o):
        return PointXYZ(self.x * o,  self.y * o, self.z * o)


class PoseXYZ():
    def __init__(self, x, y, z, R):
        self.t = PointXYZ(x, y, z)
        self.R = R
    def x_axis(self):
        x_end = self.R.dot(np.array([0.2, 0., 0.]))
        return ([self.t.x, x_end[0] + self.t.x], [self.t.y, x_end[1] + self.t.y], [self.t.z, x_end[2] + self.t.z])
    def y_axis(self):
        y_end = self.R.dot(np.array([0, 0.2, 0]))
        return ([self.t.x, y_end[0] + self.t.x], [self.t.y, y_end[1] + self.t.y], [self.t.z, y_end[2] + self.t.z])
    def z_axis(self):
        z_end = self.R.dot(np.array([0, 0, 0.2]))
        return ([self.t.x, z_end[0] + self.t.x], [self.t.y, z_end[1] + self.t.y], [self.t.z, z_end[2] + self.t.z])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_axis = ([0, 1], [0, 0], [0, 0])
y_axis = ([0, 0], [0, 1], [0, 0])
z_axis = ([0, 0], [0, 0], [0, 1])
ax.plot(x_axis[0], x_axis[1], x_axis[2], c='r')
ax.plot(y_axis[0], y_axis[1], y_axis[2], c='g')
ax.plot(z_axis[0], z_axis[1], z_axis[2], c='b')


def get_joints_poses(*, theta1, r1, theta2, r2):
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    R1 = np.array([[c1, -s1, 0],
                   [s1, c1, 0],
                   [0, 0, 1]])
    joint1_pose = PoseXYZ(0, 0, 0, R1)
    R2 = np.array([[c1, -s1*c2, s1*s2],
                   [s1, c1*c2, -c1*s2],
                   [0, s2, c2]])
    joint2_pose = PoseXYZ(r1 * c1, r1 * s1, 0, R2)
    end_effector_xyz = PointXYZ(-s1*r2*c2+r1*c1, c1*r2*c2+r1*s1, r2*s2)
    return (joint1_pose, joint2_pose, end_effector_xyz)


def get_jacobian(*, theta1, r1, theta2, r2, dof=4):
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    if dof == 2:
    # with respect to theta1, theta2
        return np.array([[-c1*r2*c2-r1*s1, s2*s1*r2],
                         [-s1*r2*c2+r1*c1, -s2*c1*r2],
                         [0, c2*r2]])
    if dof == 3:
    # with respect to theta1, theta2, r2
        return np.array([[-c1*r2*c2-r1*s1, s2*s1*r2, -s1*c2],
                         [-s1*r2*c2+r1*c1, -s2*c1*r2, c1*c2],
                         [0, c2*r2, s2]])
    if dof == 4:
    # with respect to theta1, r1, theta2, r2
        return np.array([[-c1*c2*r2-r1*s1, c1, s1*s2*r2, -s1*c2],
                         [-s1*c2*r2+r1*c1, s1, -s2*c1*r2, c1*c2],
                         [0, 0, c2*r2, s2]])
    if dof == 300:
        # with respect to theta1, r1, theta2
        return np.array([[-c1*c2*r2-r1*s1, c1, s1*s2*r2],
                         [-s1*c2*r2+r1*c1, s1, -s2*c1*r2],
                         [0, 0, c2*r2]])
    return -1


def plot_arm(*, ax, joint1_pose, joint2_pose, end_effector_xyz, color='k', end_point_size=30, line_style='-', only_end_effector =False):
    ax.scatter3D([end_effector_xyz.x], [end_effector_xyz.y],
                 [end_effector_xyz.z],
                 c=color, s=end_point_size)
    if only_end_effector:
        return
    ax.scatter3D([0, joint2_pose.t.x, end_effector_xyz.x], [0, joint2_pose.t.y, end_effector_xyz.y],
                 [0, joint2_pose.t.z, end_effector_xyz.z],
                 c='k', s=30)
    ax.plot([0, joint2_pose.t.x, end_effector_xyz.x], [0, joint2_pose.t.y, end_effector_xyz.y],
            [0, joint2_pose.t.z, end_effector_xyz.z],
                 c='k', linestyle=line_style, linewidth=3)
    def plot_joint(joint_pose):
        joint_x_axis = joint_pose.x_axis()
        joint_y_axis = joint_pose.y_axis()
        joint_z_axis = joint_pose.z_axis()
        ax.plot(joint_x_axis[0], joint_x_axis[1], joint_x_axis[2], c='r', linewidth=5)
        ax.plot(joint_y_axis[0], joint_y_axis[1], joint_y_axis[2], c='g', linewidth=5)
        ax.plot(joint_z_axis[0], joint_z_axis[1], joint_z_axis[2], c='b', linewidth=5)
    plot_joint(joint1_pose)
    plot_joint(joint2_pose)


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def get_joints_velocity(*, theta1, r1, theta2, r2, end_effector_speed, dof=4):
    # WARNING: it may not find a solution. Gauss newton method, maybe should use LM?
    step_scale = 1 # Maybe set trust region?
    max_theta_v = to_radian(2)
    max_linear_v = 1E-1

    v = [0, 0, 0, 0]
    J = get_jacobian(theta1=theta1, r1=r1, theta2=theta2, r2=r2, dof=dof)
    INF = np.transpose(J).dot(J)
    eps = 1E-4 # Solve singularity issue and make it converge faster...
    tmp = np.linalg.inv(INF + np.eye(INF.shape[0]) * eps).dot(np.transpose(J)).dot(end_effector_speed) * step_scale
    if dof == 2:
        v[0] = clamp(tmp[0], -max_theta_v, max_theta_v) # theta1
        v[2] = clamp(tmp[1], -max_theta_v, max_theta_v) # theta2
    if dof == 3:
        v[0] = clamp(tmp[0], -max_theta_v, max_theta_v) # theta1
        v[2] = clamp(tmp[1], -max_theta_v, max_theta_v) # theta2
        v[3] = clamp(tmp[2], -max_linear_v, max_linear_v) # r2
    if dof == 300:
        v[0] = clamp(tmp[0], -max_theta_v, max_theta_v) # theta1
        v[1] = clamp(tmp[1], -max_linear_v, max_linear_v) # r1
        v[2] = clamp(tmp[2], -max_theta_v, max_theta_v) # theta2
    if dof == 4:
        v[0] = clamp(tmp[0], -max_theta_v, max_theta_v) # theta1
        v[1] = clamp(tmp[1], -max_linear_v, max_linear_v) # r1
        v[2] = clamp(tmp[2], -max_theta_v, max_theta_v) # theta2
        v[3] = clamp(tmp[3], -max_linear_v, max_linear_v) # r2
    return v


def to_radian(deg):
    return deg * math.pi / 180.0




# ============ Main function starts here ===========
# I am lazy to write a main()
theta1_start = to_radian(30)
theta2_start = to_radian(60)
r1_start = 3
r2_start = 4
(joint1_pose_start, joint2_pose_start, end_effector_xyz_start) = get_joints_poses(theta1=theta1_start, r1=r1_start,
                                                             theta2=theta2_start, r2=r2_start)


theta1_goal = to_radian(60)
theta2_goal = to_radian(30)
r1_goal = 3
r2_goal = 6
(joint1_pose_goal, joint2_pose_goal, end_effector_xyz_goal) = get_joints_poses(theta1=theta1_goal, r1=r1_goal,
                                                                              theta2=theta2_goal, r2=r2_goal)


# end_effector_xyz_goal = PointXYZ(random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10))


# Test cost function
if False:
    theta1_curr = theta1_start
    r1_curr = r1_start
    theta2_curr = theta2_start
    r2_curr = r2_start
    def error(t1, t2):
        (joint1_pose_curr, joint2_pose_curr, end_effector_xyz_curr) = get_joints_poses(theta1=t1, r1=r1_curr,
                                                                                       theta2=t2, r2=r2_curr)
        err = np.linalg.norm(end_effector_xyz_goal - end_effector_xyz_curr)
        if err < 0.1:
            print(t1, t2)
        return err


    t1 = np.linspace(-math.pi, math.pi, 100)
    t2 = np.linspace(-math.pi, math.pi, 100)


    M, B = np.meshgrid(t1, t2)
    zs = np.array([error(mp, bp)
                   for mp, bp in zip(np.ravel(M), np.ravel(B))])
    Z = zs.reshape(M.shape)
    ax.plot_surface(M, B, Z, rstride=1, cstride=1, alpha=0.8, cmap=cm.coolwarm)
    plt.show()
    exit(0)


plot_arm(ax=ax, joint1_pose=joint1_pose_start, joint2_pose=joint2_pose_start,
         end_effector_xyz=end_effector_xyz_start, color='g', end_point_size = 100)


plot_arm(ax=ax, joint1_pose=joint1_pose_goal, joint2_pose=joint2_pose_goal,
         end_effector_xyz=end_effector_xyz_goal, color='r', end_point_size = 100, only_end_effector =True)


theta1_curr = theta1_start
r1_curr = r1_start
theta2_curr = theta2_start
r2_curr = r2_start
end_effector_xyz_curr = end_effector_xyz_start


# Test Jacobian
# def test_jacobian(*, theta1, r1, theta2, r2):
#     J = get_jacobian(theta1=theta1, r1=r1, theta2=theta2, r2=r2, dof=4)
#     print(J)
#     get_joints_poses(theta1=theta1, r1=r1, theta2=theta2, r2=r2)
#     small = 1E-6
#     (a, b, d_theta1) = get_joints_poses(theta1=theta1_curr+small, r1=r1_curr, theta2=theta2_curr, r2=r2_curr)
#     print((d_theta1 - end_effector_xyz_curr) / small)
#     (a, b, d_r1) = get_joints_poses(theta1=theta1_curr, r1=r1_curr+small, theta2=theta2_curr, r2=r2_curr)
#     print((d_r1 - end_effector_xyz_curr) / small)
#     (a, b, d_theta2) = get_joints_poses(theta1=theta1_curr, r1=r1_curr, theta2=theta2_curr+small, r2=r2_curr)
#     print((d_theta2 - end_effector_xyz_curr) / small)
#     (a, b, d_r2) = get_joints_poses(theta1=theta1_curr, r1=r1_curr, theta2=theta2_curr, r2=r2_curr+small)
#     print((d_r2 - end_effector_xyz_curr) / small)
# test_jacobian(theta1=theta1_curr, r1=r1_curr, theta2=theta2_curr, r2=r2_curr)


k = 0
iters = 500
termination_eps = 1E-3
dof = 4
end_effector_traj = ([end_effector_xyz_curr.x], [end_effector_xyz_curr.y], [end_effector_xyz_curr.z])
use_geometry = True
# use_geometry = False


if use_geometry:
    r = math.sqrt(end_effector_xyz_goal.x ** 2 + end_effector_xyz_goal.y ** 2)
    if dof == 3:
        assert(r1_curr < r)
        r2_curr = math.sqrt(end_effector_xyz_goal.norm() ** 2 - r1_curr ** 2)
        dof = 2
    if dof == 4:
        while r <= r1_curr:
            r1_curr /= 2.
        r2_curr = math.sqrt(end_effector_xyz_goal.norm() ** 2 - r1_curr ** 2)
        dof = 2
    (joint1_pose_curr, joint2_pose_curr, end_effector_xyz_curr) = get_joints_poses(theta1=theta1_curr, r1=r1_curr,
                                                                                   theta2=theta2_curr, r2=r2_curr)
    end_effector_traj[0].append(end_effector_xyz_curr.x)
    end_effector_traj[1].append(end_effector_xyz_curr.y)
    end_effector_traj[2].append(end_effector_xyz_curr.z)
    plot_arm(ax=ax, joint1_pose=joint1_pose_curr, joint2_pose=joint2_pose_curr,
             end_effector_xyz=end_effector_xyz_curr, color='y', line_style='-', only_end_effector =True)




while k <= iters:
    v_xyz = end_effector_xyz_goal - end_effector_xyz_curr
    v = get_joints_velocity(theta1=theta1_curr, r1=r1_curr,
                            theta2=theta2_curr, r2=r2_curr,
                            end_effector_speed=v_xyz, dof=dof)
    theta1_curr = np.unwrap(np.array([theta1_curr + v[0]]))
    r1_curr = r1_curr + v[1]
    theta2_curr = np.unwrap(np.array([theta2_curr + v[2]]))
    r2_curr = r2_curr + v[3]
    (joint1_pose_curr, joint2_pose_curr, end_effector_xyz_curr) = get_joints_poses(theta1=theta1_curr, r1=r1_curr,
                                                                                    theta2=theta2_curr, r2=r2_curr)
    end_effector_traj[0].append(end_effector_xyz_curr.x)
    end_effector_traj[1].append(end_effector_xyz_curr.y)
    end_effector_traj[2].append(end_effector_xyz_curr.z)
    err = np.linalg.norm(end_effector_xyz_goal - end_effector_xyz_curr)
    if err < termination_eps or k == iters:
        print(f'Finial error {err}')
        print(f'iter {k}')
        break
    k += 1
    plot_arm(ax=ax, joint1_pose=joint1_pose_curr, joint2_pose=joint2_pose_curr,
             end_effector_xyz=end_effector_xyz_curr, color='y', line_style='-', only_end_effector =True)
(joint1_pose_curr, joint2_pose_curr, end_effector_xyz_curr) = get_joints_poses(theta1=theta1_curr, r1=r1_curr,
                                                                               theta2=theta2_curr, r2=r2_curr)


ax.plot(end_effector_traj[0], end_effector_traj[1], end_effector_traj[2], c='y', linewidth=1)
plot_arm(ax=ax, joint1_pose=joint1_pose_curr, joint2_pose=joint2_pose_curr,
         end_effector_xyz=end_effector_xyz_curr, color='y', line_style='--')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(25, -80)
plt.show()
