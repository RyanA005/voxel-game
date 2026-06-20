Simple Raylib Voxel Parkour Engine Spec
Goal

Build a small C + raylib 3D voxel parkour prototype.

The game should generate a new fixed-size 16×16 parkour world every time it starts. The player is an AABB/cube character that can move, jump, fall, collide with blocks, die, and reach a goal.

This is the foundation for later replacing the physics_step() function with a neural network.

Tech

Use:

C
raylib
single main.c if possible

Optional structure:

/src
  main.c
  world.c/.h
  player.c/.h
  physics.c/.h
  logging.c/.h

But for the first prototype, one main.c is acceptable.

World

Use a fixed voxel map:

#define WORLD_X 16
#define WORLD_Y 16
#define WORLD_Z 16

typedef enum {
    VOXEL_EMPTY = 0,
    VOXEL_SOLID = 1,
    VOXEL_START = 2,
    VOXEL_GOAL  = 3,
    VOXEL_HAZARD = 4
} VoxelType;

static unsigned char world[WORLD_X][WORLD_Y][WORLD_Z];

Coordinate convention:

x = left/right
y = vertical
z = forward/back

Each voxel is a 1×1×1 cube.

The world should be regenerated on every program start.

World Generation

Generate simple parkour maps, not random noise.

Requirements

Every generated world should have:

a start platform
a goal platform
a chain of reachable platforms
some gaps
some height changes
a death plane below
Simple Generator

Use a random-walk platform path.

Pseudo:

void generate_world(void) {
    clear_world();

    int x = 2;
    int y = 2;
    int z = 2;

    place_platform(x, y, z, 3, 3);

    start_pos = (Vector3){x + 0.5f, y + 2.0f, z + 0.5f};

    for (int i = 0; i < 8; i++) {
        int dx = random choice from {-3, -2, 2, 3};
        int dz = random choice from {-3, -2, 2, 3};
        int dy = random choice from {-1, 0, 1};

        x = clamp(x + dx, 2, WORLD_X - 4);
        z = clamp(z + dz, 2, WORLD_Z - 4);
        y = clamp(y + dy, 1, 8);

        place_platform(x, y, z, random 2..4, random 2..4);
    }

    goal_pos = last platform center;
    place_goal_block(goal_pos);
}
Platform Placement
void place_platform(int cx, int y, int cz, int sx, int sz);

This should fill solid blocks:

x in cx..cx+sx
z in cz..cz+sz
at height y

Optional: make platforms one block thick.

Player

Use AABB physics.

typedef struct {
    Vector3 pos;
    Vector3 vel;

    float width;
    float height;

    int grounded;
    int dead;
    int won;
} Player;

Initial values:

player.pos = start_pos;
player.vel = (Vector3){0};
player.width = 0.6f;
player.height = 1.8f;
player.grounded = 0;
player.dead = 0;
player.won = 0;

The player’s pos should represent the center of the player body, not the feet.

Input
typedef struct {
    int forward;
    int back;
    int left;
    int right;
    int jump;
    int reset;
} InputState;

Controls:

W = forward
S = back
A = left
D = right
Space = jump
R = regenerate/reset
Esc = quit

For v0, use fixed world-relative movement:

W = negative Z
S = positive Z
A = negative X
D = positive X

No mouse look required yet.

Physics

The entire movement system should live in one replaceable function:

void physics_step(Player *p, InputState input, float dt);

This is the function that will eventually become:

void neural_physics_step(Player *p, InputState input, float dt);
Constants
#define MOVE_ACCEL 35.0f
#define MAX_SPEED 6.0f
#define FRICTION 12.0f
#define GRAVITY -30.0f
#define JUMP_SPEED 10.0f
Physics Step

Pseudo:

void physics_step(Player *p, InputState input, float dt) {
    float ax = 0.0f;
    float az = 0.0f;

    if (input.forward) az -= MOVE_ACCEL;
    if (input.back)    az += MOVE_ACCEL;
    if (input.left)    ax -= MOVE_ACCEL;
    if (input.right)   ax += MOVE_ACCEL;

    p->vel.x += ax * dt;
    p->vel.z += az * dt;

    apply_horizontal_friction(p, dt);
    clamp_horizontal_speed(p, MAX_SPEED);

    if (input.jump && p->grounded) {
        p->vel.y = JUMP_SPEED;
        p->grounded = 0;
    }

    p->vel.y += GRAVITY * dt;

    p->grounded = 0;

    move_axis(p, 0, p->vel.x * dt);
    move_axis(p, 1, p->vel.y * dt);
    move_axis(p, 2, p->vel.z * dt);
}
Axis Collision

Use simple axis-separated AABB resolution.

void move_axis(Player *p, int axis, float amount);

Process:

1. Move player along one axis.
2. Check if player AABB overlaps any solid voxel.
3. If collision:
   - undo movement incrementally or snap back
   - zero velocity on that axis
   - if axis is Y and player was moving downward, set grounded = 1

Simple implementation can just move, check collision, and revert:

float old = get_axis(p->pos, axis);
set_axis(&p->pos, axis, old + amount);

if (player_collides(p)) {
    set_axis(&p->pos, axis, old);

    if (axis == 0) p->vel.x = 0;
    if (axis == 1) {
        if (p->vel.y < 0) p->grounded = 1;
        p->vel.y = 0;
    }
    if (axis == 2) p->vel.z = 0;
}

This is not perfect, but good enough for v0.

Collision Helpers
int is_solid_voxel(int x, int y, int z);
int player_collides(Player *p);

player_collides() should compute the player AABB:

min_x = pos.x - width/2
max_x = pos.x + width/2
min_y = pos.y - height/2
max_y = pos.y + height/2
min_z = pos.z - width/2
max_z = pos.z + width/2

Then check all voxels overlapped by the AABB.

Out-of-bounds should be treated as solid for sides/bottom if useful, or empty with death plane. Simpler:

outside x/z bounds = solid wall
below y < -4 = death
above map = empty
Death and Win

Death:

if (player.pos.y < -4.0f) {
    reset_player();
}

Goal:

If player overlaps goal voxel:

player.won = 1;
generate_world();
reset_player();

For simplicity, the goal can be drawn as a green cube sitting on the final platform.

Rendering

Use raylib 3D.

Camera

Third-person fixed follow camera:

Camera3D camera = {0};

camera.position = (Vector3){
    player.pos.x + 8.0f,
    player.pos.y + 7.0f,
    player.pos.z + 8.0f
};

camera.target = player.pos;
camera.up = (Vector3){0, 1, 0};
camera.fovy = 60.0f;
camera.projection = CAMERA_PERSPECTIVE;

Update every frame.

Draw World

Loop through all voxels:

for x
  for y
    for z
      if world[x][y][z] != EMPTY:
        DrawCube(...)
        DrawCubeWires(...)

Colors:

solid = gray
start = blue
goal = green
hazard = red
player = orange/red

Draw cube at voxel center:

Vector3 center = { x + 0.5f, y + 0.5f, z + 0.5f };
DrawCube(center, 1, 1, 1, color);

Draw player:

DrawCube(player.pos, player.width, player.height, player.width, RED);
DrawCubeWires(player.pos, player.width, player.height, player.width, BLACK);

Draw basic UI text:

WASD move | Space jump | R reset
Map seed: <seed>
Main Loop
int main(void) {
    InitWindow(1280, 720, "Neural Voxel Parkour");
    SetTargetFPS(60);

    srand(time(NULL));
    generate_world();
    reset_player();

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if (dt > 1.0f / 30.0f) dt = 1.0f / 30.0f;

        InputState input = read_input();

        if (input.reset) {
            generate_world();
            reset_player();
        }

        physics_step(&player, input, dt);

        if (player.pos.y < -4.0f) {
            reset_player();
        }

        check_goal();

        update_camera();

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        draw_world();
        draw_player();
        EndMode3D();

        DrawText("WASD move | Space jump | R new map", 20, 20, 20, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
Neural Prep

Even before training, design the code so every tick can produce a training example.

Observation

Eventually log:

local voxel patch around player
player offset inside current voxel
velocity
grounded
input keys

For v0, use a 9×9×9 patch:

#define PATCH_R 4
#define PATCH_D 9

Observation fields:

voxels[9][9][9]
offset_x
offset_y
offset_z
vx
vy
vz
grounded
forward
back
left
right
jump

Target:

dx
dy
dz
next_vx
next_vy
next_vz
next_grounded

Where:

dx = after.pos.x - before.pos.x;
dy = after.pos.y - before.pos.y;
dz = after.pos.z - before.pos.z;
Required Seam

Make sure this call exists cleanly:

Player before = player;
physics_step(&player, input, dt);
Player after = player;

Later:

log_training_sample(before, input, after);
Success Criteria

The prototype is “done” when:

A new 16×16×16 voxel parkour map appears every launch
The player can move with WASD
The player can jump
The player collides with blocks
The player can fall and reset
The player can reach a goal
The map is simple but playable
The physics pass is isolated in physics_step()

Do not add:

textures
mesh chunks
lighting polish
mouse look
enemies
menus
saving/loading
complex generation

The point is to create the smallest possible real engine that can later produce training data for a neural player-physics model.