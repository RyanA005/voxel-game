#ifndef WORLD_H
#define WORLD_H

#include "common.h"

void generate_world(void);
void reset_player(void);
int check_goal(void);
int is_solid_voxel(int x, int y, int z);
int player_collides(Player *p);

extern int goal_vx, goal_vy, goal_vz;

#endif
