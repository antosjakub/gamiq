import pygame
from pygame.math import Vector2
import numpy as np


def get_circle_pos(theta):
    return np.array([np.sin(theta), np.cos(theta)])

def np_to_pygame_arr(arr):
    assert len(arr) == 2
    return Vector2(arr[0], arr[1])

# Initialize Pygame
pygame.init()
WHITE = (255,255,255)
NIGGA = (0,0,0)
RED = (255,0,0)

SCREEN_W = 1800
SCREEN_H = 600

# Set up the game window
window = pygame.display.set_mode((SCREEN_W, SCREEN_H))
window.fill(NIGGA)
pygame.display.set_caption("Cartpole")


ground_h_offset = 150
pygame.draw.line(window, WHITE, Vector2(0,SCREEN_H//2+ground_h_offset), Vector2(SCREEN_W,SCREEN_H//2+ground_h_offset), 1)

cartpole_w = 200
cartpole_h = 20

cartpole_arm_w = 300
cartpole_arm_h = 15
cartpole_init_arm_angle = 10

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # cartpole base
    pygame.draw.rect(window, WHITE, [
        SCREEN_W//2 - cartpole_w//2,
        SCREEN_H//2 - cartpole_h//2 + ground_h_offset,
        cartpole_w,
        cartpole_h
    ], 0)
    # cartpole arm
    arm_pnts = [
        #Vector2(cartpole_arm_w//2*i, cartpole_arm_h//2*j)
        Vector2(cartpole_arm_w*i, cartpole_arm_h*j)
        #for i,j in [[1,1], [1,-1], [-1,-1], [-1,1]]
        #for i,j in [[1,1], [1,0], [0,0], [0,1]]
        for i,j in [[1,1/2], [1,-1/2], [0,-1/2], [0,1/2]]
    ] 
    arm_coords = [
        Vector2(SCREEN_W//2, SCREEN_H//2 + ground_h_offset) +
        p.rotate(-90+cartpole_init_arm_angle) for p in arm_pnts
    ]
    pygame.draw.polygon(window, WHITE, arm_coords, 0)
    # cartpole center
    pygame.draw.circle(window, NIGGA, [
        SCREEN_W//2,
        SCREEN_H//2 + ground_h_offset
    ], 5, 0)



    pygame.display.update()

# Quit Pygame
pygame.quit()


