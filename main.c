#include "raylib.h"
#include "common.h"
#include "neural.h"
#include "observation.h"
#include "physics.h"
#include "sim.h"
#include "world.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static PhysicsMode physics_mode = PHYSICS_ANALYTIC;
static const char *model_path = "models/model_q8_h24_p3.bin";

static int try_load_model(void) {
    const char *candidates[] = {
        "models/model_q8_h24_p3.bin",
        "../models/model_q8_h24_p3.bin",
        "models/model_q8_h48x24_p3.bin",
        "../models/model_q8_h48x24_p3.bin",
        "models/model_q8_h64_p3.bin",
        "../models/model_q8_h64_p3.bin",
        "models/model_q8_h48_p3.bin",
        "../models/model_q8_h48_p3.bin",
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        if (neural_load_model(candidates[i])) {
            model_path = candidates[i];
            return 1;
        }
    }
    return 0;
}

static Color voxel_color(unsigned char t) {
    switch (t) {
        case VOXEL_SOLID:  return GRAY;
        case VOXEL_START:  return BLUE;
        case VOXEL_GOAL:   return GREEN;
        case VOXEL_HAZARD: return RED;
        default:           return BLANK;
    }
}

static void draw_world(void) {
    for (int x = 0; x < WORLD_X; x++)
        for (int y = 0; y < WORLD_Y; y++)
            for (int z = 0; z < WORLD_Z; z++) {
                if (world[x][y][z] == VOXEL_EMPTY) continue;
                Vector3 center = { x + 0.5f, y + 0.5f, z + 0.5f };
                Color c = voxel_color(world[x][y][z]);
                DrawCube(center, 1, 1, 1, c);
                DrawCubeWires(center, 1, 1, 1, DARKGRAY);
            }
}

static void draw_player(void) {
    Vector3 pos = { player.pos.x, player.pos.y, player.pos.z };
    DrawCube(pos, player.width, player.height, player.width, ORANGE);
    DrawCubeWires(pos, player.width, player.height, player.width, BLACK);
}

static Camera3D make_camera(void) {
    Camera3D cam = { 0 };
    cam.position = (Vector3){
        player.pos.x + 8.0f,
        player.pos.y + 7.0f,
        player.pos.z + 8.0f
    };
    cam.target = (Vector3){ player.pos.x, player.pos.y, player.pos.z };
    cam.up = (Vector3){ 0, 1, 0 };
    cam.fovy = 60.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    return cam;
}

static InputState read_input(void) {
    InputState in = { 0 };
    in.forward = IsKeyDown(KEY_W);
    in.back    = IsKeyDown(KEY_S);
    in.left    = IsKeyDown(KEY_A);
    in.right   = IsKeyDown(KEY_D);
    in.jump    = IsKeyDown(KEY_SPACE);
    in.reset   = IsKeyPressed(KEY_R);
    return in;
}

static void check_goal(void) {
    int gx = (int)goal_pos.x;
    int gy = (int)goal_pos.y;
    int gz = (int)goal_pos.z;

    float min_x = player.pos.x - player.width * 0.5f;
    float max_x = player.pos.x + player.width * 0.5f;
    float min_y = player.pos.y - player.height * 0.5f;
    float max_y = player.pos.y + player.height * 0.5f;
    float min_z = player.pos.z - player.width * 0.5f;
    float max_z = player.pos.z + player.width * 0.5f;

    if (gx + 0.5f >= min_x && gx + 0.5f <= max_x &&
        gy + 0.5f >= min_y && gy + 0.5f <= max_y &&
        gz + 0.5f >= min_z && gz + 0.5f <= max_z) {
        player.won = 1;
        generate_world();
        reset_player();
    }
}

static void print_usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s                         Interactive game (Tab toggles neural/analytic)\n", prog);
    printf("  %s --record N --out FILE    Record N training samples\n", prog);
    printf("  %s --bench MODEL            Rollout benchmark vs analytic physics\n", prog);
}

static int parse_args(int argc, char **argv, int *record_n, char **record_out, char **bench_model) {
    *record_n = 0;
    *record_out = NULL;
    *bench_model = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--record") == 0 && i + 1 < argc) {
            *record_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            *record_out = argv[++i];
        } else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            *bench_model = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int record_n = 0;
    char *record_out = NULL;
    char *bench_model = NULL;

    if (parse_args(argc, argv, &record_n, &record_out, &bench_model) != 0)
        return 0;

    if (record_n > 0 && record_out) {
        unsigned int seed = (unsigned int)time(NULL);
        return sim_record_dataset(record_out, record_n, seed);
    }

    if (bench_model) {
        return sim_benchmark(bench_model, 300, 50, 42);
    }

    InitWindow(1280, 720, "Neural Voxel Parkour");
    SetTargetFPS(60);

    map_seed = (unsigned int)time(NULL);
    srand((int)map_seed);
    generate_world();
    reset_player();

    if (try_load_model())
        physics_mode = PHYSICS_NEURAL;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) break;

        if (IsKeyPressed(KEY_TAB)) {
            physics_mode = physics_mode == PHYSICS_ANALYTIC ? PHYSICS_NEURAL : PHYSICS_ANALYTIC;
            if (physics_mode == PHYSICS_NEURAL && !neural_model_loaded())
                try_load_model();
        }

        if (IsKeyPressed(KEY_N))
            try_load_model();

        float dt = FIXED_DT;

        InputState input = read_input();

        if (input.reset) {
            map_seed = (unsigned int)time(NULL);
            srand((int)map_seed);
            generate_world();
            reset_player();
        }

        if (physics_mode == PHYSICS_ANALYTIC)
            physics_step(&player, input, dt);
        else
            neural_physics_step(&player, input, dt);

        if (player.pos.y < -4.0f)
            reset_player();

        check_goal();

        Camera3D camera = make_camera();

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        draw_world();
        draw_player();
        EndMode3D();

        DrawText("WASD move | Space jump | R new map | Tab physics | N reload model", 20, 20, 20, DARKGRAY);
        DrawText(TextFormat("Map seed: %u", map_seed), 20, 45, 20, DARKGRAY);
        DrawText(TextFormat("Physics: %s", physics_mode == PHYSICS_NEURAL ? "NEURAL" : "ANALYTIC"),
                 20, 70, 20, physics_mode == PHYSICS_NEURAL ? GREEN : DARKGRAY);
        DrawText(TextFormat("Model: %s", model_path), 20, 95, 20, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
