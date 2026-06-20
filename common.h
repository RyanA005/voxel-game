#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

#define WORLD_X 16
#define WORLD_Y 16
#define WORLD_Z 16

#define FIXED_DT (1.0f / 60.0f)

#define MOVE_ACCEL 35.0f
#define MAX_SPEED 6.0f
#define FRICTION 12.0f
#define GRAVITY -30.0f
#define JUMP_SPEED 10.0f

#define PATCH_R 1
#define PATCH_D 3
#define VOXEL_COUNT (PATCH_D * PATCH_D * PATCH_D)

#define INPUT_DIM (VOXEL_COUNT + 12)
#define OUTPUT_DIM 7

#define DATASET_MAGIC 0x4B435056u /* VPCK */
#define MODEL_MAGIC 0x214D4C50u   /* MLP! */
#define MODEL_VER_FLOAT 1
#define MODEL_VER_INT8  2

typedef enum {
    VOXEL_EMPTY = 0,
    VOXEL_SOLID = 1,
    VOXEL_START = 2,
    VOXEL_GOAL = 3,
    VOXEL_HAZARD = 4
} VoxelType;

typedef struct { float x, y, z; } Vec3;

typedef struct {
    Vec3 pos;
    Vec3 vel;
    float width;
    float height;
    int grounded;
    int dead;
    int won;
} Player;

typedef struct {
    int forward;
    int back;
    int left;
    int right;
    int jump;
    int reset;
} InputState;

typedef struct {
    unsigned char voxels[PATCH_D][PATCH_D][PATCH_D];
    float offset_x, offset_y, offset_z;
    float vx, vy, vz;
    int grounded;
    int forward, back, left, right, jump;
} Observation;

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t record_size;
} DatasetHeader;

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct {
    unsigned char voxels[VOXEL_COUNT];
    float offset[3];
    float vel[3];
    uint8_t grounded;
    uint8_t input[5];
    float dt;
    float target_delta[3];
    float target_vel[3];
    uint8_t target_grounded;
    uint32_t seed;
}
#ifdef __GNUC__
__attribute__((packed))
#endif
TrainingRecord;
#ifdef _MSC_VER
#pragma pack(pop)
#endif

typedef enum { PHYSICS_ANALYTIC, PHYSICS_NEURAL } PhysicsMode;

extern unsigned char world[WORLD_X][WORLD_Y][WORLD_Z];
extern Player player;
extern Vec3 start_pos;
extern Vec3 goal_pos;
extern unsigned int map_seed;

#endif
