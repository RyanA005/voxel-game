#ifndef NEURAL_H
#define NEURAL_H

#include "common.h"

int neural_load_model(const char *path);
int neural_require_model(const char *path);
int neural_model_loaded(void);
int neural_quant_type(void);
const char *neural_quant_name(void);
const char *neural_kernel_name(void);
void neural_forward(const float *input_raw, float *output_raw);
void neural_physics_step(Player *p, InputState input, float dt);
double neural_forward_only_us(const float *input_raw, float *output_raw, int iterations);

#endif
