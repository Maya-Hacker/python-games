import sys
import math

# Grab Snaffles and try to throw them through the opponent's goal!
# Move towards a Snaffle and use your team id to determine where you need to throw it.

my_team_id = int(input())  # if 0 you need to score on the right of the map, if 1 you need to score on the left

class Entity:
    def __init__(self, inputs: list[str]):
        self.id = int(inputs[0])  # entity identifier
        self.entity_type = inputs[1]  # "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" (or "BLUDGER" after first league)
        self.x = int(inputs[2])  # position
        self.y = int(inputs[3])  # position
        self.vx = int(inputs[4])  # velocity
        self.vy = int(inputs[5])  # velocity

class Snaffle(Entity):
    def __init__(self, inputs: list[str]):
        super().__init__(inputs)
        self.is_held = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise

class Wizard(Entity):
    def __init__(self, inputs: list[str]):
        super().__init__(inputs)
        self.has_snaffle = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise
        self.do_the_thing = ""
        self.thrust = 150

    def throw(self):
        if my_team_id == 0:
            goal_x = 16000
        else:
            goal_x = 0
        self.do_the_thing = f"THROW {goal_x} 3750 500"

    def move_to_snaffle(self, snaffleindex: int):
        the_snaffle = the_list_of_snaffles[snaffleindex]
        self.do_the_thing = f"MOVE {the_snaffle.x} {the_snaffle.y} {self.thrust}"

    
the_list_of_snaffles: list[Snaffle] = []


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

    harry: Wizard = None
    ron: Wizard = None
    draco: Wizard = None
    crab: Wizard = None

    the_list_of_snaffles = []

    for i in range(entities):
        inputs = input().split()
        print(f"{inputs=}", file=sys.stderr, flush=True)
        
        entity_id = int(inputs[0])  # entity identifier
        entity_type = inputs[1]  # "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" (or "BLUDGER" after first league)

        if entity_type == 'WIZARD':
            if not harry:
                harry = Wizard(inputs)
            else:
                ron = Wizard(inputs)
        if entity_type == 'OPPONENT_WIZARD':
            if not draco:
                draco = Wizard(inputs)
            else:
                crab = Wizard(inputs)

        if entity_type == 'SNAFFLE':
            the_list_of_snaffles.append(Snaffle(inputs))

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

    if harry.has_snaffle:
        harry.throw()
    else:
        harry.move_to_snaffle(0)

    if ron.has_snaffle:
        ron.throw()
    else:
        ron.move_to_snaffle(1)


    print(harry.do_the_thing)
    print(ron.do_the_thing)