#include "observation.h"
#include "world.h"
#include <math.h>

static int g_patch_n = PATCH_MAX;

void obs_set_patch_n(int patch_n) {
    if (patch_n < 2) patch_n = 2;
    if (patch_n > PATCH_MAX) patch_n = PATCH_MAX;
    g_patch_n = patch_n;
}

int obs_get_patch_n(void) { return g_patch_n; }

int obs_input_dim(void) { return patch_input_dim(g_patch_n); }

static int input_is_idle(InputState input) {
    return !input.forward && !input.back && !input.left && !input.right && !input.jump;
}

static void sanitize_grounded_idle_targets(const Player *before, InputState input, TrainingRecord *rec) {
    if (!input_is_idle(input) || !before->grounded || !rec->target_grounded)
        return;

    float dy = rec->target_delta[1];
    float vy = rec->target_vel[1];
    float horiz_d = fabsf(rec->target_delta[0]) + fabsf(rec->target_delta[2]);
    float horiz_v = fabsf(rec->target_vel[0]) + fabsf(rec->target_vel[2]);

    /* v3: only snap the tiniest micro-bounces; preserve real fall/edge motion. */
    if (fabsf(dy) < 0.0003f && fabsf(vy) < 0.05f && horiz_d < 0.0003f && horiz_v < 0.05f) {
        rec->target_delta[0] = rec->target_delta[1] = rec->target_delta[2] = 0.0f;
        rec->target_vel[0] = rec->target_vel[1] = rec->target_vel[2] = 0.0f;
        rec->target_grounded = 1;
    }
}

void build_observation(const Player *p, InputState input, Observation *obs) {
    int cx = (int)p->pos.x;
    int cy = (int)p->pos.y;
    int cz = (int)p->pos.z;
    int n = g_patch_n;
    int d0 = -(n / 2);

    for (int ix = 0; ix < n; ix++)
        for (int iy = 0; iy < n; iy++)
            for (int iz = 0; iz < n; iz++) {
                int dx = d0 + ix;
                int dy = d0 + iy;
                int dz = d0 + iz;
                int wx = cx + dx, wy = cy + dy, wz = cz + dz;
                unsigned char v = VOXEL_EMPTY;
                if (wx >= 0 && wx < WORLD_X && wy >= 0 && wy < WORLD_Y && wz >= 0 && wz < WORLD_Z)
                    v = world[wx][wy][wz];
                else if (wx < 0 || wx >= WORLD_X || wz < 0 || wz >= WORLD_Z)
                    v = VOXEL_SOLID;
                obs->voxels[ix][iy][iz] = v;
            }

    obs->offset_x = p->pos.x - cx;
    obs->offset_y = p->pos.y - cy;
    obs->offset_z = p->pos.z - cz;
    obs->vx = p->vel.x;
    obs->vy = p->vel.y;
    obs->vz = p->vel.z;
    obs->grounded = p->grounded;
    obs->forward = input.forward;
    obs->back = input.back;
    obs->left = input.left;
    obs->right = input.right;
    obs->jump = input.jump;
}

void pack_observation(const Observation *obs, float *out) {
    int n = g_patch_n;
    int i = 0;
    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            for (int z = 0; z < n; z++)
                out[i++] = obs->voxels[x][y][z] / 4.0f;

    out[i++] = obs->offset_x;
    out[i++] = obs->offset_y;
    out[i++] = obs->offset_z;
    out[i++] = obs->vx / MAX_SPEED;
    out[i++] = obs->vy / JUMP_SPEED;
    out[i++] = obs->vz / MAX_SPEED;
    out[i++] = obs->grounded ? 1.0f : 0.0f;
    out[i++] = obs->forward ? 1.0f : 0.0f;
    out[i++] = obs->back ? 1.0f : 0.0f;
    out[i++] = obs->left ? 1.0f : 0.0f;
    out[i++] = obs->right ? 1.0f : 0.0f;
    out[i++] = obs->jump ? 1.0f : 0.0f;
}

void make_training_record(const Player *before, InputState input, const Player *after, TrainingRecord *rec) {
    int saved = g_patch_n;
    obs_set_patch_n(PATCH_MAX);

    Observation obs;
    build_observation(before, input, &obs);

    int vi = 0;
    for (int x = 0; x < PATCH_MAX; x++)
        for (int y = 0; y < PATCH_MAX; y++)
            for (int z = 0; z < PATCH_MAX; z++)
                rec->voxels[vi++] = obs.voxels[x][y][z];

    obs_set_patch_n(saved);

    rec->offset[0] = obs.offset_x;
    rec->offset[1] = obs.offset_y;
    rec->offset[2] = obs.offset_z;
    rec->vel[0] = before->vel.x;
    rec->vel[1] = before->vel.y;
    rec->vel[2] = before->vel.z;
    rec->grounded = (uint8_t)before->grounded;
    rec->input[0] = (uint8_t)input.forward;
    rec->input[1] = (uint8_t)input.back;
    rec->input[2] = (uint8_t)input.left;
    rec->input[3] = (uint8_t)input.right;
    rec->input[4] = (uint8_t)input.jump;
    rec->dt = FIXED_DT;
    rec->target_delta[0] = after->pos.x - before->pos.x;
    rec->target_delta[1] = after->pos.y - before->pos.y;
    rec->target_delta[2] = after->pos.z - before->pos.z;
    rec->target_vel[0] = after->vel.x;
    rec->target_vel[1] = after->vel.y;
    rec->target_vel[2] = after->vel.z;
    rec->target_grounded = (uint8_t)after->grounded;
    rec->seed = map_seed;
    rec->pos_before[0] = before->pos.x;
    rec->pos_before[1] = before->pos.y;
    rec->pos_before[2] = before->pos.z;
    sanitize_grounded_idle_targets(before, input, rec);
}
