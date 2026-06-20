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

#define PATCH_MAX 9
#define PATCH_R_MAX 4
#define INPUT_EXTRA 12
#define INPUT_DIM_MAX (PATCH_MAX * PATCH_MAX * PATCH_MAX + INPUT_EXTRA)
#define OUTPUT_DIM 7

/* Legacy defaults (full 9³ patch). */
#define PATCH_R 4
#define PATCH_D 9
#define INPUT_DIM 741
#define HIDDEN1 128
#define HIDDEN2 128

static inline int patch_input_dim(int patch_n) {
    return patch_n * patch_n * patch_n + INPUT_EXTRA;
}

static inline int patch_from_input_dim(int input_dim) {
    int v = input_dim - INPUT_EXTRA;
    for (int n = 2; n <= PATCH_MAX; n++)
        if (n * n * n == v) return n;
    return PATCH_MAX;
}

#define DATASET_MAGIC 0x4B435056u /* VPCK */
#define DATASET_VERSION 2
#define DATASET_RECORD_SIZE 804
#define MODEL_MAGIC 0x214D4C50u   /* MLP! */
#define MODEL_VERSION_FP32 1u
#define MODEL_VERSION_QUANT 2u

/* Quantized weight schemes (model version 2). Activations stay FP32 (W8A32 / W4A32). */
#define QUANT_INT8_ROW   1u  /* per output-row symmetric int8 + float scale */
#define QUANT_INT8_LAYER 2u  /* single scale per weight tensor */
#define QUANT_INT4_ROW   3u  /* per output-row symmetric int4 (packed) + float scale */
#define QUANT_FP16       4u  /* IEEE half weights; expanded to FP32 at load */

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
    unsigned char voxels[PATCH_MAX][PATCH_MAX][PATCH_MAX];
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
    unsigned char voxels[729];
    float offset[3];
    float vel[3];
    uint8_t grounded;
    uint8_t input[5];
    float dt;
    float target_delta[3];
    float target_vel[3];
    uint8_t target_grounded;
    uint32_t seed;
    float pos_before[3];
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
