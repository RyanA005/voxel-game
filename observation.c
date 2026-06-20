#include "observation.h"
#include "world.h"

void build_observation(const Player *p, InputState input, Observation *obs) {
    int cx = (int)p->pos.x;
    int cy = (int)p->pos.y;
    int cz = (int)p->pos.z;

    for (int dx = -PATCH_R; dx <= PATCH_R; dx++)
        for (int dy = -PATCH_R; dy <= PATCH_R; dy++)
            for (int dz = -PATCH_R; dz <= PATCH_R; dz++) {
                int wx = cx + dx, wy = cy + dy, wz = cz + dz;
                unsigned char v = VOXEL_EMPTY;
                if (wx >= 0 && wx < WORLD_X && wy >= 0 && wy < WORLD_Y && wz >= 0 && wz < WORLD_Z)
                    v = world[wx][wy][wz];
                else if (wx < 0 || wx >= WORLD_X || wz < 0 || wz >= WORLD_Z)
                    v = VOXEL_SOLID;
                obs->voxels[dx + PATCH_R][dy + PATCH_R][dz + PATCH_R] = v;
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
    int i = 0;
    for (int x = 0; x < PATCH_D; x++)
        for (int y = 0; y < PATCH_D; y++)
            for (int z = 0; z < PATCH_D; z++)
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

void pack_player_features(const Player *p, InputState input, float *out) {
    int cx = (int)p->pos.x;
    int cy = (int)p->pos.y;
    int cz = (int)p->pos.z;
    int oi = 0;

    for (int dx = -PATCH_R; dx <= PATCH_R; dx++)
        for (int dy = -PATCH_R; dy <= PATCH_R; dy++)
            for (int dz = -PATCH_R; dz <= PATCH_R; dz++) {
                int wx = cx + dx, wy = cy + dy, wz = cz + dz;
                unsigned char v = VOXEL_EMPTY;
                if ((unsigned)wx < (unsigned)WORLD_X &&
                    (unsigned)wy < (unsigned)WORLD_Y &&
                    (unsigned)wz < (unsigned)WORLD_Z)
                    v = world[wx][wy][wz];
                else if (wx < 0 || wx >= WORLD_X || wz < 0 || wz >= WORLD_Z)
                    v = VOXEL_SOLID;
                out[oi++] = v * 0.25f;
            }

    out[VOXEL_COUNT + 0] = p->pos.x - (float)cx;
    out[VOXEL_COUNT + 1] = p->pos.y - (float)cy;
    out[VOXEL_COUNT + 2] = p->pos.z - (float)cz;
    out[VOXEL_COUNT + 3] = p->vel.x / MAX_SPEED;
    out[VOXEL_COUNT + 4] = p->vel.y / JUMP_SPEED;
    out[VOXEL_COUNT + 5] = p->vel.z / MAX_SPEED;
    out[VOXEL_COUNT + 6] = p->grounded ? 1.0f : 0.0f;
    out[VOXEL_COUNT + 7] = input.forward ? 1.0f : 0.0f;
    out[VOXEL_COUNT + 8] = input.back ? 1.0f : 0.0f;
    out[VOXEL_COUNT + 9] = input.left ? 1.0f : 0.0f;
    out[VOXEL_COUNT + 10] = input.right ? 1.0f : 0.0f;
    out[VOXEL_COUNT + 11] = input.jump ? 1.0f : 0.0f;
}

void make_training_record(const Player *before, InputState input, const Player *after, TrainingRecord *rec) {
    Observation obs;
    build_observation(before, input, &obs);

    int vi = 0;
    for (int x = 0; x < PATCH_D; x++)
        for (int y = 0; y < PATCH_D; y++)
            for (int z = 0; z < PATCH_D; z++)
                rec->voxels[vi++] = obs.voxels[x][y][z];

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
}
