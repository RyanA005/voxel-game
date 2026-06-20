"""Python port of C world/physics/observation for closed-loop training."""

from __future__ import annotations

import numpy as np

WORLD_X = WORLD_Y = WORLD_Z = 16
FIXED_DT = 1.0 / 60.0
MOVE_ACCEL = 35.0
MAX_SPEED = 6.0
FRICTION = 12.0
GRAVITY = -30.0
JUMP_SPEED = 10.0
PATCH_D = 9
PATCH_R = 4
PLAYER_HEIGHT = 1.8
PLAYER_WIDTH = 0.6

VOXEL_EMPTY = 0
VOXEL_SOLID = 1
VOXEL_START = 2
VOXEL_GOAL = 3


class CRand:
    """MSVC/MinGW-compatible rand() for matching C world generation."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def rand(self) -> int:
        self.state = (self.state * 214013 + 2531011) & 0xFFFFFFFF
        return (self.state >> 16) & 0x7FFF

    def rand_range(self, lo: int, hi: int) -> int:
        return lo + self.rand() % (hi - lo + 1)

    def rand_choice(self, opts):
        return opts[self.rand() % len(opts)]


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def generate_world(seed: int):
    rng = CRand(seed)
    world = np.zeros((WORLD_X, WORLD_Y, WORLD_Z), dtype=np.uint8)

    def place(cx, y, cz, sx, sz, vtype):
        for x in range(cx, cx + sx + 1):
            for z in range(cz, cz + sz + 1):
                if 0 <= x < WORLD_X and 0 <= z < WORLD_Z and 0 <= y < WORLD_Y:
                    world[x, y, z] = vtype

    x, y, z = 2, 2, 2
    place(x, y, z, 3, 3, VOXEL_START)
    top = float(y + 1)
    start_pos = np.array([x + 1.5, top + PLAYER_HEIGHT * 0.5 + 0.05, z + 1.5], dtype=np.float32)

    last_sx = last_sz = 3
    for _ in range(8):
        dx = rng.rand_choice([-3, -2, 2, 3])
        dz = rng.rand_choice([-3, -2, 2, 3])
        dy = rng.rand_choice([-1, 0, 1])
        x = clamp_int(x + dx, 2, WORLD_X - 4)
        z = clamp_int(z + dz, 2, WORLD_Z - 4)
        y = clamp_int(y + dy, 1, 8)
        last_sx = rng.rand_range(2, 4)
        last_sz = rng.rand_range(2, 4)
        place(x, y, z, last_sx, last_sz, VOXEL_SOLID)

    gx = x + last_sx // 2
    gy = y
    gz = z + last_sz // 2
    if 0 <= gx < WORLD_X and 0 <= gy < WORLD_Y and 0 <= gz < WORLD_Z:
        world[gx, gy, gz] = VOXEL_GOAL

    return world, start_pos


def is_solid(world, x: int, y: int, z: int) -> bool:
    if x < 0 or x >= WORLD_X or z < 0 or z >= WORLD_Z:
        return True
    if y < 0 or y >= WORLD_Y:
        return False
    t = int(world[x, y, z])
    return t in (VOXEL_SOLID, VOXEL_START, VOXEL_GOAL)


def player_collides(world, pos, width=PLAYER_WIDTH, height=PLAYER_HEIGHT) -> bool:
    min_x = pos[0] - width * 0.5
    max_x = pos[0] + width * 0.5
    min_y = pos[1] - height * 0.5
    max_y = pos[1] + height * 0.5
    min_z = pos[2] - width * 0.5
    max_z = pos[2] + width * 0.5
    ix0, ix1 = int(min_x), int(max_x)
    iy0, iy1 = int(min_y), int(max_y)
    iz0, iz1 = int(min_z), int(max_z)
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            for iz in range(iz0, iz1 + 1):
                if is_solid(world, ix, iy, iz):
                    return True
    return False


def init_state_from_pos_vel(pos, vel, grounded):
    return {
        "pos": np.asarray(pos, dtype=np.float32).copy(),
        "vel": np.asarray(vel, dtype=np.float32).copy(),
        "grounded": int(grounded),
    }


def copy_state(state):
    return {
        "pos": state["pos"].copy(),
        "vel": state["vel"].copy(),
        "grounded": int(state["grounded"]),
    }


def teacher_state_vector(state):
    return np.concatenate([state["pos"], state["vel"], [float(state["grounded"])]])


def _move_axis(world, state, axis: int, amount: float):
    pos = state["pos"]
    vel = state["vel"]
    old = float(pos[axis])
    pos[axis] = old + amount
    if player_collides(world, pos):
        pos[axis] = old
        if axis == 0:
            vel[0] = 0.0
        elif axis == 1:
            if vel[1] < 0:
                state["grounded"] = 1
            vel[1] = 0.0
        else:
            vel[2] = 0.0


def physics_step(world, state, inp):
    pos = state["pos"]
    vel = state["vel"]
    ax = az = 0.0
    if inp.get("forward"):
        az -= MOVE_ACCEL
    if inp.get("back"):
        az += MOVE_ACCEL
    if inp.get("left"):
        ax -= MOVE_ACCEL
    if inp.get("right"):
        ax += MOVE_ACCEL

    vel[0] += ax * FIXED_DT
    vel[2] += az * FIXED_DT
    vel[0] -= vel[0] * FRICTION * FIXED_DT
    vel[2] -= vel[2] * FRICTION * FIXED_DT
    hs = float(np.sqrt(vel[0] ** 2 + vel[2] ** 2))
    if hs > MAX_SPEED:
        s = MAX_SPEED / hs
        vel[0] *= s
        vel[2] *= s

    if inp.get("jump") and state["grounded"]:
        vel[1] = JUMP_SPEED
        state["grounded"] = 0

    vel[1] += GRAVITY * FIXED_DT
    state["grounded"] = 0
    _move_axis(world, state, 0, vel[0] * FIXED_DT)
    _move_axis(world, state, 1, vel[1] * FIXED_DT)
    _move_axis(world, state, 2, vel[2] * FIXED_DT)


def apply_neural_step(state, out):
    state["pos"][0] += out[0]
    state["pos"][1] += out[1]
    state["pos"][2] += out[2]
    state["vel"][0] = out[3]
    state["vel"][1] = out[4]
    state["vel"][2] = out[5]
    state["grounded"] = 1 if out[6] >= 0.5 else 0


def pack_observation(world, pos, vel, grounded, inp) -> np.ndarray:
    cx, cy, cz = int(pos[0]), int(pos[1]), int(pos[2])
    n = PATCH_D
    d0 = -(n // 2)
    voxels = np.zeros(n * n * n, dtype=np.float32)
    i = 0
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                wx = cx + d0 + ix
                wy = cy + d0 + iy
                wz = cz + d0 + iz
                if 0 <= wx < WORLD_X and 0 <= wy < WORLD_Y and 0 <= wz < WORLD_Z:
                    v = int(world[wx, wy, wz])
                elif wx < 0 or wx >= WORLD_X or wz < 0 or wz >= WORLD_Z:
                    v = VOXEL_SOLID
                else:
                    v = VOXEL_EMPTY
                voxels[i] = v / 4.0
                i += 1

    off = pos - np.array([cx, cy, cz], dtype=np.float32)
    return np.concatenate(
        [
            voxels,
            off,
            [vel[0] / MAX_SPEED, vel[1] / JUMP_SPEED, vel[2] / MAX_SPEED],
            [float(grounded)],
            [float(inp.get("forward", 0)), float(inp.get("back", 0)),
             float(inp.get("left", 0)), float(inp.get("right", 0)), float(inp.get("jump", 0))],
        ]
    ).astype(np.float32)


def settle(world, state, frames=60):
    none = {"forward": 0, "back": 0, "left": 0, "right": 0, "jump": 0}
    for _ in range(frames):
        physics_step(world, state, none)
