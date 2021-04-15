from mass_spring_robot_config import robots
import random
import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os

real = ti.f64
ti.init(default_fp=real)

max_steps = 4096
vis_interval = 256

steps = 2048 // 2
output_vis_interval = steps // 120

assert steps * 2 <= max_steps

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

x = vec()
v = vec()
v_inc = vec()

head_id = 6

n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -1.8
friction = 2.5

damping = 1

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

center = vec()


@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_inc)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation)
    ti.root.dense(ti.i, max_steps).place(center)
    ti.root.lazy_grad()


dt = 0.004


@ti.kernel
def compute_center(t: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, i]
        center[t] = (1.0 / n_objects) * c


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm(1e-8) + 1e-4

        target_length = spring_length[i]
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        # Dashpot damping
        x_ij = x[t, a] - x[t, b]
        d = x_ij.normalized()
        v_rel = (v[t, a] - v[t, b]).dot(d)
        impulse += 1. * v_rel * d

        ti.atomic_add(v_inc[t + 1, a], -impulse)
        ti.atomic_add(v_inc[t + 1, b], impulse)


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        # s = math.exp(-dt * damping)
        s = 1.
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Vector([0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, i] = new_v
        x[t, i] = new_x


gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF)


def forward(output="run", visualize=True):
    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)

    total_steps = steps if not output else steps * 2

    for t in range(1, total_steps):
        print(x.to_numpy())
        compute_center(t - 1)
        apply_spring_force(t - 1)
        advance_toi(t)
        if (t + 1) % interval == 0 and visualize:
            gui.clear()
            gui.line((0, ground_height), (1, ground_height),
                     color=0x0,
                     radius=3)

            def circle(x, y, color):
                gui.circle((x, y), ti.rgb_to_hex(color), 7)

            for i in range(n_springs):

                def get_pt(x):
                    return (x[0], x[1])

                a = 0 * 0.5
                r = 2
                if spring_actuation[i] == 0:
                    a = 0
                    c = 0x222222
                else:
                    r = 4
                    c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                gui.line(get_pt(x[t, spring_anchor_a[i]]),
                         get_pt(x[t, spring_anchor_b[i]]),
                         color=c,
                         radius=r)

            for i in range(n_objects):
                color = (0.4, 0.6, 0.6)
                if i == head_id:
                    color = (0.8, 0.2, 0.3)
                circle(x[t, i][0], x[t, i][1], color)

            if output:
                gui.show('mass_spring/{}/{:04d}.png'.format(output, t))
            else:
                gui.show()


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            x.grad[t, i] = ti.Vector([0.0, 0.0])
            v.grad[t, i] = ti.Vector([0.0, 0.0])
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            v_inc.grad[t, i] = ti.Vector([0.0, 0.0])


def clear():
    clear_states()


def setup_robot(objects, springs):
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = [objects[i][0] + 0.4, objects[i][1] + 0.5]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3] / 4
        spring_actuation[i] = s[4]


robot_id = 0
if len(sys.argv) != 2:
    print("Usage: python3 mass_spring_interactive.py [robot_id=0, 1, 2, ...]")
    exit(0)
else:
    robot_id = int(sys.argv[1])

if __name__ == '__main__':
    setup_robot(*robots[robot_id]())

    forward()
