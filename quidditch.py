import sys
import math

# Grab Snaffles and try to throw them through the opponent's goal!
# Move towards a Snaffle and use your team id to determine where you need to throw it.

my_team_id = int(input())  # if 0 you need to score on the right of the map, if 1 you need to score on the left

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{self.x} {self.y}"

    def dist(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

class Velocity(Position):
    pass


class Entity:
    def __init__(self, inputs: list[str]):
        self.id = int(inputs[0])  # entity identifier
        self.entity_type = inputs[1]  # "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" (or "BLUDGER" after first league)
        self.pos = Position(int(inputs[2]), int(inputs[3]))
        self.vel = Velocity(int(inputs[4]), int(inputs[5]))

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

    def throw(self, pos: Position | None = None):
        if not pos:
            if my_team_id == 0:
                pos = goal_pos_R
            else:
                pos = goal_pos_L

        self.do_the_thing = f"THROW {pos} 500"

    def move_to_snaffle(self, snaffleindex: int):
        # print(f"move_to_snaffle {snaffleindex=}", file=sys.stderr, flush=True)
        the_snaffle = the_list_of_snaffles[snaffleindex]
        self.do_the_thing = f"MOVE {the_snaffle.pos} {self.thrust}"

    def protect_rings(self):
        prot_pos = Position(PROTECTION_RING_DIST, 3750)
        if my_team_id == 0:
            pass
        else:
            prot_pos.x = 16000 - PROTECTION_RING_DIST
        dist_to_protect = self.pos.dist(prot_pos)
        thrust = int(min(self.thrust, dist_to_protect * SLOW_DOWN_FACTOR))
        self.do_the_thing = f"MOVE {prot_pos} {thrust}"

    def obstruction(self):
        return False

    
the_list_of_snaffles: list[Snaffle] = []
goal_pos_R = Position(16000, 3750)
goal_pos_L = Position(0, 3750)
# example
goal_pos_R.x
print(f"{goal_pos_R=}, {goal_pos_R.dist(goal_pos_L)}", file=sys.stderr, flush=True)
# fff

SLOW_DOWN_FACTOR = 4
PROTECTION_RING_DIST = 3000

# game loop
while True:
    my_score, my_magic = [int(i) for i in input().split()]
    opponent_score, opponent_magic = [int(i) for i in input().split()]
    entities = int(input())  # number of entities still in game

    harry: Wizard = None
    ron: Wizard = None
    draco: Wizard = None
    crabbe: Wizard = None

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
                crabbe = Wizard(inputs)

        if entity_type == 'SNAFFLE':
            the_list_of_snaffles.append(Snaffle(inputs))

    if harry.has_snaffle:
        harry.throw()
    else:
        harry.move_to_snaffle(0)

    if ron.has_snaffle:
        if ron.obstruction():
            ron.throw([harry.x, harry.y])
        ron.throw()
    elif len(the_list_of_snaffles) > 1:
        ron.move_to_snaffle(1)
    else:
        ron.protect_rings()



    print(harry.do_the_thing)
    print(ron.do_the_thing)