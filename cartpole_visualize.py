import pygame
from pygame.math import Vector2
import numpy as np
import cartpole_physics


# Initialize Pygame
pygame.init()
WHITE = (255,255,255)
NIGGA = (0,0,0)
RED = (255,0,0)

SCREEN_W = 1800
SCREEN_H = 600

# Set up the game window
window = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Cartpole")
clock = pygame.time.Clock()


ground_h_offset = 150

cartpole_w = 200
cartpole_h = 20

cartpole_arm_w = 300
cartpole_arm_h = 15

# initial conditions
cartpole = cartpole_physics.CartPole(
    pole_length=cartpole_arm_w,
    x=0,
    x_dot=0,
    theta=cartpole_physics.rad_from_deg(0),
    theta_dot=0
)

i = 0
# Game loop
running = True
while running:
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        #print("Left arrow is being held down.")
        action = -1
    if keys[pygame.K_RIGHT]:
        #print("Right arrow is being held down.")
        action = 1

    dt = clock.tick(100)
    delta_t = dt/500
    cartpole.step(delta_t, action)

    
    # GRAPHICS
    pygame.display.flip()
    window.fill(NIGGA)
    pygame.draw.line(window, WHITE, Vector2(0,SCREEN_H//2+ground_h_offset), Vector2(SCREEN_W,SCREEN_H//2+ground_h_offset), 1)
    # -- cartpole base
    pygame.draw.rect(window, WHITE, [
        SCREEN_W//2 - cartpole_w//2 + cartpole.x,
        SCREEN_H//2 - cartpole_h//2 + ground_h_offset,
        cartpole_w,
        cartpole_h
    ], 0)
    # -- cartpole arm
    arm_pnts = [
        Vector2(cartpole_arm_w*i, cartpole_arm_h*j)
        for i,j in [[1,1/2], [1,-1/2], [0,-1/2], [0,1/2]]
    ] 
    arm_coords = [
        Vector2(SCREEN_W//2 + cartpole.x, SCREEN_H//2 + ground_h_offset) +
        p.rotate(-90+cartpole_physics.deg_from_rad(cartpole.theta)) for p in arm_pnts
    ]
    pygame.draw.polygon(window, WHITE, arm_coords, 0)
    # -- cartpole center
    pygame.draw.circle(window, NIGGA, [
        SCREEN_W//2 + cartpole.x,
        SCREEN_H//2 + ground_h_offset
    ], 5, 0)

    pygame.display.update()

    i+=1
    #if i > 100:
    #    break

# Quit Pygame
pygame.quit()


