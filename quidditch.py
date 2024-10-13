import sys
import math

# Grab Snaffles and try to throw them through the opponent's goal!
# Move towards a Snaffle and use your team id to determine where you need to throw it.

my_team_id = int(input())  # if 0 you need to score on the right of the map, if 1 you need to score on the left

# game loop
while True:
    my_score, my_magic = [int(i) for i in input().split()]
    opponent_score, opponent_magic = [int(i) for i in input().split()]
    entities = int(input())  # number of entities still in game

    s4_x = 0
    s4_y = 0
    w0_s = 0
    s5_x = 0
    s5_y = 0
    w1_s = 0

    for i in range(entities):
        inputs = input().split()
        print(f"{inputs=}", file=sys.stderr, flush=True)

        entity_id = int(inputs[0])  # entity identifier
        entity_type = inputs[1]  # "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" (or "BLUDGER" after first league)
        x = int(inputs[2])  # position
        y = int(inputs[3])  # position
        vx = int(inputs[4])  # velocity
        vy = int(inputs[5])  # velocity
        state = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise

        if entity_id == 0:
            w0_s = state 
        if entity_id == 1:
            w1_s = state
        if entity_id == 4:
            s4_x = x
            s4_y = y
        if entity_id == 5:
            s5_x = x
            s5_y = y 

    print(f"{w0_s=}", file=sys.stderr, flush=True)
    print(f"{w1_s=}", file=sys.stderr, flush=True)

    for i in range(2):
        go_x = 0
        go_y = 0
        thrust = 150
        throw_instead = False
        if i == 0:
            go_x = s4_x
            go_y = s4_y
            if w0_s:
                throw_instead = True
 
        if i == 1:
            go_x = s5_x
            go_y = s5_y
            if w1_s:
                throw_instead = True



        #do it !!!!
        if throw_instead:
            print("THROW 16000 3750 500")
        else:
            print(f"MOVE {go_x} {go_y} {thrust}")
