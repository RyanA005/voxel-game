#include "common.h"
#include <stdlib.h>

unsigned char world[WORLD_X][WORLD_Y][WORLD_Z];
Player player;
Vec3 start_pos;
Vec3 goal_pos;
int goal_vx, goal_vy, goal_vz;
unsigned int map_seed;

#define PLAYER_HEIGHT 1.8f
#define PLAYER_WIDTH 0.6f

static int player_collides_impl(Player *p);

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int rand_range(int lo, int hi) {
    return lo + rand() % (hi - lo + 1);
}

static int rand_choice(const int *opts, int n) {
    return opts[rand() % n];
}

static void clear_world(void) {
    for (int x = 0; x < WORLD_X; x++)
        for (int y = 0; y < WORLD_Y; y++)
            for (int z = 0; z < WORLD_Z; z++)
                world[x][y][z] = VOXEL_EMPTY;
}

static void place_platform(int cx, int y, int cz, int sx, int sz, VoxelType type) {
    for (int x = cx; x <= cx + sx; x++)
        for (int z = cz; z <= cz + sz; z++)
            if (x >= 0 && x < WORLD_X && z >= 0 && z < WORLD_Z && y >= 0 && y < WORLD_Y)
                world[x][y][z] = type;
}

void generate_world(void) {
    clear_world();

    int x = 2, y = 2, z = 2;
    int last_sx = 3, last_sz = 3;

    place_platform(x, y, z, 3, 3, VOXEL_START);
    {
        float top = (float)(y + 1);
        start_pos = (Vec3){
            x + 1.5f,
            top + PLAYER_HEIGHT * 0.5f + 0.05f,
            z + 1.5f
        };
    }

    for (int i = 0; i < 8; i++) {
        int dx_opts[] = { -3, -2, 2, 3 };
        int dz_opts[] = { -3, -2, 2, 3 };
        int dy_opts[] = { -1, 0, 1 };
        int dx = rand_choice(dx_opts, 4);
        int dz = rand_choice(dz_opts, 4);
        int dy = rand_choice(dy_opts, 3);

        x = clamp_int(x + dx, 2, WORLD_X - 4);
        z = clamp_int(z + dz, 2, WORLD_Z - 4);
        y = clamp_int(y + dy, 1, 8);

        last_sx = rand_range(2, 4);
        last_sz = rand_range(2, 4);
        place_platform(x, y, z, last_sx, last_sz, VOXEL_SOLID);
    }

    goal_vx = x + last_sx / 2;
    goal_vy = y;
    goal_vz = z + last_sz / 2;
    if (goal_vx >= 0 && goal_vx < WORLD_X && goal_vy >= 0 && goal_vy < WORLD_Y &&
        goal_vz >= 0 && goal_vz < WORLD_Z)
        world[goal_vx][goal_vy][goal_vz] = VOXEL_GOAL;
    goal_pos = (Vec3){ goal_vx + 0.5f, goal_vy + 0.5f, goal_vz + 0.5f };
}

void reset_player(void) {
    player.vel = (Vec3){ 0 };
    player.width = PLAYER_WIDTH;
    player.height = PLAYER_HEIGHT;
    player.pos = start_pos;
    player.grounded = 0;
    player.dead = 0;
    player.won = 0;
    for (int i = 0; i < 64 && player_collides_impl(&player); i++)
        player.pos.y += 0.1f;
}

static int player_touches_goal(void) {
    float pmin_x = player.pos.x - player.width * 0.5f;
    float pmax_x = player.pos.x + player.width * 0.5f;
    float pmin_y = player.pos.y - player.height * 0.5f;
    float pmax_y = player.pos.y + player.height * 0.5f;
    float pmin_z = player.pos.z - player.width * 0.5f;
    float pmax_z = player.pos.z + player.width * 0.5f;

    float gx0 = (float)goal_vx;
    float gx1 = gx0 + 1.0f;
    float gy0 = (float)goal_vy;
    float gy1 = gy0 + 1.0f;
    float gz0 = (float)goal_vz;
    float gz1 = gz0 + 1.0f;

    int xz_overlap = pmin_x < gx1 && pmax_x > gx0 && pmin_z < gz1 && pmax_z > gz0;
    if (!xz_overlap) return 0;

    /* Inside or against the voxel (jump through, walk into side). */
    if (pmin_y < gy1 && pmax_y > gy0) return 1;

    /* Standing on top: feet on the goal block surface. */
    float feet_y = pmin_y;
    float goal_top = gy1;
    if (feet_y >= goal_top - 0.25f && feet_y <= goal_top + 0.35f) return 1;

    return 0;
}

int check_goal(void) {
    if (player_touches_goal()) {
        player.won = 1;
        generate_world();
        reset_player();
        return 1;
    }
    return 0;
}

int is_solid_voxel(int x, int y, int z) {
    if (x < 0 || x >= WORLD_X || z < 0 || z >= WORLD_Z) return 1;
    if (y < 0 || y >= WORLD_Y) return 0;
    VoxelType t = (VoxelType)world[x][y][z];
    return t == VOXEL_SOLID || t == VOXEL_START || t == VOXEL_GOAL || t == VOXEL_HAZARD;
}

int player_collides(Player *p) {
    return player_collides_impl(p);
}

static int player_collides_impl(Player *p) {
    float min_x = p->pos.x - p->width * 0.5f;
    float max_x = p->pos.x + p->width * 0.5f;
    float min_y = p->pos.y - p->height * 0.5f;
    float max_y = p->pos.y + p->height * 0.5f;
    float min_z = p->pos.z - p->width * 0.5f;
    float max_z = p->pos.z + p->width * 0.5f;

    int ix0 = (int)min_x, ix1 = (int)max_x;
    int iy0 = (int)min_y, iy1 = (int)max_y;
    int iz0 = (int)min_z, iz1 = (int)max_z;

    for (int ix = ix0; ix <= ix1; ix++)
        for (int iy = iy0; iy <= iy1; iy++)
            for (int iz = iz0; iz <= iz1; iz++)
                if (is_solid_voxel(ix, iy, iz)) return 1;
    return 0;
}
