#ifndef OBSERVATION_H
#define OBSERVATION_H

#include "common.h"

void build_observation(const Player *p, InputState input, Observation *obs);
void pack_observation(const Observation *obs, float *out);
void pack_player_features(const Player *p, InputState input, float *out);
void make_training_record(const Player *before, InputState input, const Player *after, TrainingRecord *rec);

#endif
