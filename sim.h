#ifndef SIM_H
#define SIM_H

int sim_record_dataset(const char *out_path, int num_samples, unsigned int base_seed);
int sim_benchmark(const char *model_path, int steps_per_episode, int num_episodes, unsigned int base_seed);

#endif
