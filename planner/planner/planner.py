#!/usr/bin/env python3
import matplotlib.pyplot as plt
import math
import copy
import numpy as np

from util import doIntersect, minDistance

#
#
#
#-------------|
# o     |     |
#-------------|

kLeftBoundaryLine = [(0, 3), (15, 3)]
kRightBoundaryLine = [(0, -3), (15, -3)]
kObstacleLine = [[(6, -0.3), (6, 0.3)],
                 [(10, -3), (10, -2)],
                 [(10, 3), (10, 2)]]

# kLeftBoundaryLine = [(0, 3), (5, 3)]
# kRightBoundaryLine = [(0, -3), (5, -3)]
# kObstacleLine = []
QuickMode = False
MaxNumOfDrawPath = 2000
# Tree search depth
kSteeringLimit = 0.05
kSteeringStep = 0.01
kPlanSteps = 8
kDepth = 6
kPlanDist = 5.0
kTargetLine = [kRightBoundaryLine[-1], kLeftBoundaryLine[-1]]

# front wheel to back wheel distance
kBaseline = 0.5
car_length = 1
car_width = 0.2

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class State:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def inverse(self):
        s = math.sin(self.t)
        c = math.cos(self.t)
        return State(-s*self.y-c*self.x, -c*self.y + s*self.x, -self.t)

    def compose(self, s2):
        s = math.sin(self.t)
        c = math.cos(self.t)
        return State(c*s2.x - s*s2.y + self.x, s*s2.x + c*s2.y + self.y, self.t + s2.t)

    def transform(self, p):
        s = math.sin(self.t)
        c = math.cos(self.t)
        return Point(c*p.x - s*p.y + self.x, s*p.x + c*p.y + self.y)


# Positive theta turn left
def bicycle(distance, theta):
    # Or we can let vehicle go straight
    minitantheta = 0.0001
    tantheta = math.tan(theta)
    go_straight = False
    if tantheta >= 0 and tantheta < minitantheta:
        go_straight = True
    if tantheta < 0 and tantheta > -minitantheta:
        go_straight = True
    if go_straight:
        return State(distance, 0, 0)
    r = kBaseline / tantheta
    phi = distance / r
    if (tantheta == minitantheta):
        return State(distance, 0, 0)
    return State(r*math.sin(phi), r*(1-math.cos(phi)), phi)


# Given moving distance and steering change, return clothoid path in nsteps.
def clothoid(nsteps, distance, theta0, theta1, start_xyt, max_step = -1):
    path = [start_xyt]
    dxyt = State(0, 0, 0)
    for step in range(0, nsteps):
        if step == max_step:
            break
        alpha = 1.0 * step / (nsteps-1)
        step_dxyt = bicycle(distance / nsteps, theta0 *
                            (1 - alpha) + theta1 * alpha)
        new_dxyt = dxyt.compose(step_dxyt)
        path.append(start_xyt.compose(new_dxyt))
        dxyt = new_dxyt
    return path


# Check whether the car runs into a wall
def check_collision(state_xyt):
    def car_intersects_a_line(p, q):
        wheel_to_bumper = (car_length-kBaseline)/2
        v_lb = state_xyt.transform(Point(-wheel_to_bumper, car_width/2))
        v_rb = state_xyt.transform(Point(-wheel_to_bumper, -car_width/2))
        v_lt = state_xyt.transform(Point(kBaseline+wheel_to_bumper, car_width/2))
        v_rt = state_xyt.transform(Point(kBaseline+wheel_to_bumper, -car_width/2))
        return doIntersect(p, q, v_lb, v_lt) or doIntersect(p, q, v_lt, v_rt) \
            or doIntersect(p, q, v_rt, v_rb) or doIntersect(p, q, v_rb, v_lb)

    def check_collision_to_boundary(boundary_line):
        for i in range(0, len(boundary_line)-1):
            p = Point(boundary_line[i][0], boundary_line[i][1])
            q = Point(boundary_line[i+1][0], boundary_line[i+1][1])
            if car_intersects_a_line(p, q):
                return True

    for obs in kObstacleLine:
        if car_intersects_a_line(Point(obs[0][0], obs[0][1]), Point(obs[1][0], obs[1][1])):
            return True

    a = check_collision_to_boundary(kLeftBoundaryLine)
    b = check_collision_to_boundary(kRightBoundaryLine)
    return a or b


# If the car pass the finish line.
def check_reach_goal(state_xyt):
    wheel_to_bumper = (car_length-kBaseline)/2
    p = Point(kTargetLine[0][0], kTargetLine[0][1])
    q = Point(kTargetLine[1][0], kTargetLine[1][1])
    wheel_to_tail = (car_length-kBaseline)/2
    v_lb = state_xyt.transform(Point(-wheel_to_bumper, car_width/2))
    v_rb = state_xyt.transform(Point(-wheel_to_bumper, -car_width/2))
    v_lt = state_xyt.transform(Point(kBaseline+wheel_to_bumper, car_width/2))
    v_rt = state_xyt.transform(Point(kBaseline+wheel_to_bumper, -car_width/2))
    return doIntersect(p, q, v_lb, v_lt) or doIntersect(p, q, v_lt, v_rt) \
        or doIntersect(p, q, v_rt, v_rb) or doIntersect(p, q, v_rb, v_lb)



# Define a cost function, keep the car away from boundaries
def cal_cost(state_xyt):
    return abs(state_xyt.y)
    E = (state_xyt.x, state_xyt.y)
    def min_dist_to_boundary(boundary_line):
        min_dist = 1000000000000.0
        for i in range(0, len(boundary_line)-1):
            dist = minDistance(boundary_line[i], boundary_line[i+1], E)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    d_l = min_dist_to_boundary(kLeftBoundaryLine)
    d_r = min_dist_to_boundary(kRightBoundaryLine)
    return abs(abs(d_r) - abs(d_l))


class Vplan:
    def __init__(self):
        self.step_plans = {}
        # smaller, better
        self.scores = {}
        self.success = False


class StepPlan:
    def __init__(self, xyt, theta, steps):
        self.xyt = xyt
        self.theta = theta
        self.steps = steps

best_plan = Vplan()
best_plan.scores[0] = 1000000000000
all_plans = []

def plan_r(this_plan, depth, all_plans):
    global best_plan
    if depth == kDepth:
        # we failed to search a path, but if we haven't found one, we will copy it
        if best_plan.success == False:
            keys = best_plan.scores.keys()
            if this_plan.scores[kDepth-1] < best_plan.scores[max(keys)]:
                best_plan = copy.deepcopy(this_plan)
        failed_plan = Vplan()
        failed_plan.success = True
        for key in this_plan.step_plans:
            if key <= depth+1:
                failed_plan.step_plans[key] = copy.deepcopy(
                    this_plan.step_plans[key])
                failed_plan.scores[key] = copy.deepcopy(
                    this_plan.scores[key])
        # Failed plans
        all_plans.append(copy.deepcopy(failed_plan))
        return False

    xyt0 = copy.deepcopy(this_plan.step_plans[depth].xyt)
    xyt1 = copy.deepcopy(xyt0)
    nsteps = kPlanSteps
    theta_range = list(np.linspace(-kSteeringLimit, kSteeringLimit, int(2*kSteeringLimit/kSteeringStep)))
    # [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3] # kSteeringLimit,
    theta_range.append(0.0)
    for theta in theta_range:
        cost = 0
        dxyt_all = State(0, 0, 0)
        has_collision = False;
        reached_goal = False;
        for step in range(0, nsteps):
            alpha = 1.0 * step / (nsteps-1)
            steering_angle = this_plan.step_plans[depth].theta * (
                1 - alpha) + theta * alpha

            dxyt = bicycle(kPlanDist / nsteps, steering_angle)
            dxyt_all = dxyt_all.compose(dxyt)
            xyt1 = xyt0.compose(dxyt_all)
            cost += cal_cost(xyt1)
            this_plan.scores[depth+1] = this_plan.scores[depth] + cost
            if depth+1 in this_plan.step_plans:
                # overwrite existing path
                this_plan.step_plans[depth+1].theta = theta
                this_plan.step_plans[depth+1].xyt = copy.deepcopy(xyt1)
                this_plan.step_plans[depth+1].steps = step+1
            else:
                this_plan.step_plans[depth+1] = StepPlan(xyt1, theta, step+1)

            if check_collision(xyt1):
                has_collision = True
                break

            if check_reach_goal(xyt1):
                # success
                keys = best_plan.scores.keys()
                if this_plan.scores[depth+1] < best_plan.scores[max(keys)]:
                    best_plan = Vplan()
                    best_plan.success = True
                    for key in this_plan.step_plans:
                        if key <= depth+1:
                            best_plan.step_plans[key] = copy.deepcopy(
                                this_plan.step_plans[key])
                            best_plan.scores[key] = copy.deepcopy(
                                this_plan.scores[key])
                success_plan = Vplan()
                success_plan.success = True
                for key in this_plan.step_plans:
                    if key <= depth+1:
                        success_plan.step_plans[key] = copy.deepcopy(
                            this_plan.step_plans[key])
                        success_plan.scores[key] = copy.deepcopy(
                            this_plan.scores[key])
                all_plans.append(copy.deepcopy(success_plan))
                reached_goal = True
                break

        if reached_goal:
            continue

        if has_collision:
            failed_plan = Vplan()
            failed_plan.success = True
            for key in this_plan.step_plans:
                if key <= depth+1:
                    failed_plan.step_plans[key] = copy.deepcopy(
                        this_plan.step_plans[key])
                    failed_plan.scores[key] = copy.deepcopy(
                        this_plan.scores[key])
            # Failed plans
            all_plans.append(copy.deepcopy(failed_plan))
            continue

        if plan_r(this_plan, depth + 1, all_plans) and QuickMode:
            return True
    return False

def plan(start_xyt, all_plans):
    this_plan = Vplan()
    this_plan.step_plans[0] = StepPlan(start_xyt, 0, 0)
    this_plan.scores[0] = 0
    plan_r(this_plan, 0, all_plans)


def plot_path(path, all_paths):
    plt.plot([elem[0] for elem in kRightBoundaryLine],
             [elem[1] for elem in kRightBoundaryLine], 'b')
    plt.plot([elem[0] for elem in kLeftBoundaryLine],
             [elem[1] for elem in kLeftBoundaryLine], 'b')
    plt.plot([kTargetLine[0][0], kTargetLine[1][0]], [kTargetLine[0][1], kTargetLine[1][1]], 'g')

    for obs in kObstacleLine:
        plt.plot([e[0] for e in obs], [e[1] for e in obs], 'k')

    plt.pause(1)
    print("Draw all paths, ", len(all_paths))
    for i, a_path in enumerate(all_paths):
        if i == MaxNumOfDrawPath:
            break
        plt.plot([state_xyt.x for state_xyt in a_path[0]], [state_xyt.y for state_xyt in a_path[0]], '-*')
        # plt.pause(5)

    for state_xyt in path:
        x = []
        y = []
        wheel_to_bumper = (car_length-kBaseline)/2
        v_lb = state_xyt.transform(Point(-wheel_to_bumper, car_width/2))
        v_rb = state_xyt.transform(Point(-wheel_to_bumper, -car_width/2))
        v_lt = state_xyt.transform(Point(kBaseline+wheel_to_bumper, car_width/2))
        v_rt = state_xyt.transform(Point(kBaseline+wheel_to_bumper, -car_width/2))
        x.extend([v_lb.x, v_lt.x, v_rt.x, v_rb.x, v_lb.x])
        y.extend([v_lb.y, v_lt.y, v_rt.y, v_rb.y, v_lb.y])
        plt.plot(x, y, 'r')
        v1 = state_xyt.transform(Point(kBaseline+wheel_to_bumper, 0))
        v2 = state_xyt.transform(Point(kBaseline+wheel_to_bumper+0.1, 0))
        plt.plot([v1.x, v2.x], [v1.y, v2.y], 'k')
        plt.pause(0.01)
    plt.show()


def main():
    global best_plan
    max_iter = 10
    start_xyt = State(0, 0, 0)
    for i in range(max_iter):
        all_plans = []
        selected_path = []
        print("iter ", i)
        best_plan = Vplan()
        best_plan.scores[0] = 1000000000000
        plan(start_xyt, all_plans)
        print("[=====]", best_plan.success)
        #print("path: ======")
        #for key in best_plan.step_plans:
        #    print(key, best_plan.step_plans[key].xyt.x,
        #          best_plan.step_plans[key].xyt.y, best_plan.step_plans[key].xyt.t)
        #print("cost: ======")
        #for key in best_plan.scores:
        #    print(best_plan.scores[key])
        keys = best_plan.step_plans.keys()
        keys = sorted(keys)
        for i, key in enumerate(keys):
            if i == len(keys)-1:
                break
            selected_path.extend(clothoid(kPlanSteps, kPlanDist,
                                 best_plan.step_plans[key].theta, best_plan.step_plans[key+1].theta,
                                 best_plan.step_plans[key].xyt, best_plan.step_plans[key+1].steps))
        all_paths = []
        for a_plan in all_plans:
            keys = a_plan.step_plans.keys()
            keys = sorted(keys)
            path = []
            for i, key in enumerate(keys):
                if i == len(keys)-1:
                    break
                path.extend(clothoid(kPlanSteps, kPlanDist,
                                     a_plan.step_plans[key].theta, a_plan.step_plans[key+1].theta,
                                     a_plan.step_plans[key].xyt, a_plan.step_plans[key+1].steps))
            if len(path):
                all_paths.append((path, a_plan.scores[max(keys)]))

        plot_path(selected_path, all_paths)
        if best_plan.success:
            break
        start_xyt = selected_path[-1]


if __name__ == "__main__":
    main()
