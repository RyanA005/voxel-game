#include "physics.h"
#include "world.h"
#include <math.h>

static float get_axis(Vec3 v, int axis) {
    if (axis == 0) return v.x;
    if (axis == 1) return v.y;
    return v.z;
}

static void set_axis(Vec3 *v, int axis, float val) {
    if (axis == 0) v->x = val;
    else if (axis == 1) v->y = val;
    else v->z = val;
}

static void move_axis(Player *p, int axis, float amount) {
    float old = get_axis(p->pos, axis);
    set_axis(&p->pos, axis, old + amount);

    if (player_collides(p)) {
        set_axis(&p->pos, axis, old);
        if (axis == 0) p->vel.x = 0;
        else if (axis == 1) {
            if (p->vel.y < 0) p->grounded = 1;
            p->vel.y = 0;
        } else p->vel.z = 0;
    }
}

static void apply_horizontal_friction(Player *p, float dt) {
    p->vel.x -= p->vel.x * FRICTION * dt;
    p->vel.z -= p->vel.z * FRICTION * dt;
}

static void clamp_horizontal_speed(Player *p, float max_speed) {
    float speed = sqrtf(p->vel.x * p->vel.x + p->vel.z * p->vel.z);
    if (speed > max_speed) {
        float s = max_speed / speed;
        p->vel.x *= s;
        p->vel.z *= s;
    }
}

void physics_step(Player *p, InputState input, float dt) {
    float ax = 0.0f, az = 0.0f;

    if (input.forward) az -= MOVE_ACCEL;
    if (input.back)    az += MOVE_ACCEL;
    if (input.left)    ax -= MOVE_ACCEL;
    if (input.right)   ax += MOVE_ACCEL;

    p->vel.x += ax * dt;
    p->vel.z += az * dt;

    apply_horizontal_friction(p, dt);
    clamp_horizontal_speed(p, MAX_SPEED);

    if (input.jump && p->grounded) {
        p->vel.y = JUMP_SPEED;
        p->grounded = 0;
    }

    p->vel.y += GRAVITY * dt;
    p->grounded = 0;

    move_axis(p, 0, p->vel.x * dt);
    move_axis(p, 1, p->vel.y * dt);
    move_axis(p, 2, p->vel.z * dt);
}
