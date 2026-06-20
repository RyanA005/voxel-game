#ifndef OBSERVATION_H
#define OBSERVATION_H

#include "common.h"

void obs_set_patch_n(int patch_n);
int obs_get_patch_n(void);
int obs_input_dim(void);

void build_observation(const Player *p, InputState input, Observation *obs);
void pack_observation(const Observation *obs, float *out);
void make_training_record(const Player *before, InputState input, const Player *after, TrainingRecord *rec);

#endif
