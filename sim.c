#include "sim.h"
#include "neural.h"
#include "observation.h"
#include "physics.h"
#include "world.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct {
    int hold_frames;
    int policy;
    int edge_dir;
    InputState cur;
} InputPolicy;

static InputState empty_input(void) {
    InputState in = { 0 };
    return in;
}

static void set_edge_dir(InputState *in, int dir) {
    *in = empty_input();
    if (dir == 0) in->forward = 1;
    else if (dir == 1) in->back = 1;
    else if (dir == 2) in->left = 1;
    else in->right = 1;
}

static InputState random_input_policy(InputPolicy *pol) {
    if (pol->hold_frames <= 0) {
        pol->cur = empty_input();

        /* 10% idle when grounded (reduced from v2). */
        if (player.grounded && (rand() % 100) < 10) {
            pol->policy = -1;
            pol->hold_frames = 20 + rand() % 41;
            return pol->cur;
        }

        /* 40% fall / drift while airborne. */
        if (!player.grounded && (rand() % 100) < 40) {
            pol->policy = -3;
            pol->hold_frames = 8 + rand() % 25;
            if (rand() % 4 == 0) {
                int dir = rand() % 4;
                set_edge_dir(&pol->cur, dir);
            }
            return pol->cur;
        }

        pol->hold_frames = 3 + rand() % 13;
        pol->policy = rand() % 100;

        if (pol->policy < 10) {
            pol->hold_frames = 20 + rand() % 41;
        } else if (pol->policy < 25) {
            /* Walk toward platform edge ~15%. */
            pol->policy = -2;
            pol->edge_dir = rand() % 4;
            pol->hold_frames = 20 + rand() % 61;
            set_edge_dir(&pol->cur, pol->edge_dir);
        } else if (pol->policy < 55) {
            pol->cur.forward = rand() % 2;
            pol->cur.back = rand() % 2;
            pol->cur.left = rand() % 2;
            pol->cur.right = rand() % 2;
            pol->cur.jump = rand() % 5 == 0;
        } else if (pol->policy < 80) {
            int dir = rand() % 4;
            set_edge_dir(&pol->cur, dir);
            pol->cur.jump = rand() % 8 == 0;
        } else if (pol->policy < 95) {
            pol->cur.forward = 1;
            pol->cur.jump = rand() % 3 == 0;
        }
    }

    if (pol->policy == -2) {
        set_edge_dir(&pol->cur, pol->edge_dir);
    }

    pol->hold_frames--;
    return pol->cur;
}

static void settle_player(int frames) {
    InputState none = empty_input();
    for (int i = 0; i < frames; i++)
        physics_step(&player, none, FIXED_DT);
}

static int record_is_idle_grounded(const TrainingRecord *rec) {
    int idle = rec->input[0] == 0 && rec->input[1] == 0 && rec->input[2] == 0 &&
               rec->input[3] == 0 && rec->input[4] == 0;
    return idle && rec->grounded && rec->target_grounded;
}

static double time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, c;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec * 1e-3;
#endif
}

static int record_is_airborne(const TrainingRecord *rec) {
    return !rec->grounded;
}

int sim_record_dataset(const char *out_path, int num_samples, unsigned int base_seed) {
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", out_path);
        return 1;
    }

    DatasetHeader hdr = { DATASET_MAGIC, DATASET_VERSION, (uint16_t)sizeof(TrainingRecord) };
    fwrite(&hdr, sizeof(hdr), 1, f);

    InputPolicy pol = { 0, 0, 0, empty_input() };
    int episode_steps = 0;
    int written = 0;

    map_seed = base_seed;
    srand((int)map_seed);
    generate_world();
    reset_player();
    settle_player(60);

    int idle_grounded_count = 0;
    int airborne_count = 0;

    while (written < num_samples) {
        InputState input = random_input_policy(&pol);

        Player before = player;
        physics_step(&player, input, FIXED_DT);
        Player after = player;

        TrainingRecord rec;
        make_training_record(&before, input, &after, &rec);
        if (record_is_idle_grounded(&rec)) idle_grounded_count++;
        if (record_is_airborne(&rec)) airborne_count++;
        fwrite(&rec, sizeof(rec), 1, f);
        written++;
        episode_steps++;

        if (player.pos.y < -4.0f || episode_steps >= 400) {
            map_seed = base_seed + (unsigned int)written;
            srand((int)map_seed);
            generate_world();
            reset_player();
            settle_player(60);
            pol.hold_frames = 0;
            episode_steps = 0;
        }
    }

    fclose(f);
    printf("Recorded %d samples to %s\n", written, out_path);
    printf("Idle+grounded samples: %d (%.1f%%)\n", idle_grounded_count,
           100.0f * idle_grounded_count / (written > 0 ? written : 1));
    printf("Airborne samples: %d (%.1f%%)\n", airborne_count,
           100.0f * airborne_count / (written > 0 ? written : 1));
    return 0;
}

int sim_benchmark(const char *model_path, int steps_per_episode, int num_episodes, unsigned int base_seed) {
    (void)model_path;

    double sum_pos_err = 0.0, sum_vel_err = 0.0;
    int grounded_mismatch = 0, total_steps = 0;
    int tunnel_analytic = 0, tunnel_neural = 0;
    double t_analytic = 0.0, t_neural = 0.0;

    for (int ep = 0; ep < num_episodes; ep++) {
        unsigned int seed = base_seed + (unsigned int)ep * 7919u;
        map_seed = seed;
        srand((int)seed);
        generate_world();
        reset_player();

        InputPolicy pol = { 0, 0, 0, empty_input() };
        Player analytic = player;
        Player neural_p = player;

        for (int step = 0; step < steps_per_episode; step++) {
            InputState input = random_input_policy(&pol);

            double t0 = time_us();
            physics_step(&analytic, input, FIXED_DT);
            t_analytic += time_us() - t0;

            t0 = time_us();
            neural_physics_step(&neural_p, input, FIXED_DT);
            t_neural += time_us() - t0;

            float dx = analytic.pos.x - neural_p.pos.x;
            float dy = analytic.pos.y - neural_p.pos.y;
            float dz = analytic.pos.z - neural_p.pos.z;
            sum_pos_err += sqrtf(dx * dx + dy * dy + dz * dz);

            float dvx = analytic.vel.x - neural_p.vel.x;
            float dvy = analytic.vel.y - neural_p.vel.y;
            float dvz = analytic.vel.z - neural_p.vel.z;
            sum_vel_err += sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);

            if (analytic.grounded != neural_p.grounded) grounded_mismatch++;

            player = analytic;
            if (player_collides(&analytic) && !analytic.grounded) tunnel_analytic++;
            player = neural_p;
            if (player_collides(&neural_p) && !neural_p.grounded) tunnel_neural++;

            total_steps++;
            if (analytic.pos.y < -4.0f || neural_p.pos.y < -4.0f) break;
        }
    }

    printf("=== Rollout Benchmark (%d episodes x up to %d steps) ===\n", num_episodes, steps_per_episode);
    printf("Mean position error:  %.5f\n", sum_pos_err / total_steps);
    printf("Mean velocity error:  %.5f\n", sum_vel_err / total_steps);
    printf("Grounded mismatch:    %.2f%%\n", 100.0 * grounded_mismatch / total_steps);
    printf("Tunnel (analytic):    %d / %d steps (non-grounded overlap)\n", tunnel_analytic, total_steps);
    printf("Tunnel (neural):      %d / %d steps (non-grounded overlap)\n", tunnel_neural, total_steps);
    printf("Avg analytic step:    %.3f us\n", t_analytic / total_steps);
    printf("Avg neural step:      %.3f us\n", t_neural / total_steps);
    printf("Inference kernel:     %s\n", neural_kernel_name());
    printf("Model quant:          %s\n", neural_quant_name());

    float sample_in[INPUT_DIM_MAX] = { 0 };
    float sample_out[OUTPUT_DIM] = { 0 };
    sample_in[patch_input_dim(obs_get_patch_n()) - INPUT_EXTRA] = 0.5f;
    double fwd_only = neural_forward_only_us(sample_in, sample_out, 10000);
    printf("Forward-only (10k avg): %.3f us\n", fwd_only);

    return 0;
}
