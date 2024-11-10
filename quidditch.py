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

    def distance_to_goal(self):
        return self.pos.x - goal_attack.x

    def velocity_to_goal(self) -> float:
        if my_team_id == 1:
            return -self.vel.x
        return self.vel.x

class Snaffle(Entity):
    def __init__(self, inputs: list[str]):
        super().__init__(inputs)
        self.is_held = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise

    def __repr__(self):
        return f"Snaffle({self.pos})"

class Wizard(Entity):
    def __init__(self, inputs: list[str]):
        super().__init__(inputs)
        self.has_snaffle = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise
        self.do_the_thing = ""
        self.thrust = 150

    def throw(self, pos: Position | None = None):
        if not pos:
            pos = goal_attack
        self.do_the_thing = f"THROW {pos} 500"

    def closest_snaffle(self):
        def dist_to_self(snaffle):
            return self.pos.dist(snaffle.pos)
        # print(f"{the_list_of_snaffles=}", file=sys.stderr, flush=True)
        closest_snaffles = sorted(the_list_of_snaffles, key=dist_to_self)
        # print(f"{closest_snaffles=}", file=sys.stderr, flush=True)
        the_snaffle = closest_snaffles[0]
        return the_snaffle

    def move_to_snaffle(self):
        the_snaffle = self.closest_snaffle()
        self.do_the_thing = f"MOVE {the_snaffle.pos} {self.thrust}"

    def protect_rings(self):
        prot_pos = Position(PROTECTION_RING_DIST, 3750)
        if my_team_id == 0:
            pass
        else:
            prot_pos.x = 16000 - PROTECTION_RING_DIST

        snaf = self.closest_snaffle()
        prot_pos.y = snaf.pos.y + snaf.vel.y * 5
        prot_pos.y = min(UPPER_POST_Y - WIS_RAD, max(LOWER_POST_Y + WIS_RAD, prot_pos.y))


        dist_to_protect = self.pos.dist(prot_pos)
        thrust = int(min(self.thrust, dist_to_protect * SLOW_DOWN_FACTOR))
        self.do_the_thing = f"MOVE {prot_pos} {thrust}"

    def obliviate(self, bludger):
        self.do_the_thing = f"OBLIVIATE {bludger.id}"

    def petrificus(self, entity):
        self.do_the_thing = f"PETRIFICUS {entity.id}"

    def accio(self, snaffle):
        self.do_the_thing = f"ACCIO {snaffle.id}"

    def flipendo(self, entity):
        self.do_the_thing = f"FLIPENDO {entity.id}"

    def obstruction(self):
        return False



WIS_RAD = 400
the_list_of_snaffles: list[Snaffle] = []
goal_pos_R = Position(16000, 3750)
goal_pos_L = Position(0, 3750)
UPPER_POST_Y = 3750+2000
LOWER_POST_Y = 3750-2000
goal_attack = goal_pos_R
goal_defend = goal_pos_L
if my_team_id == 1:
    goal_attack = goal_pos_L
    goal_defend = goal_pos_R


SLOW_DOWN_FACTOR = 4
PROTECTION_RING_DIST = 1500

FLIPENDO_COST = 20

def snaffle_close_to_goal():
    for snaffle in the_list_of_snaffles:
        if snaffle.distance_to_goal() < 1500:
            return snaffle


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
        harry.move_to_snaffle()

    if ron.has_snaffle:
        if ron.obstruction():
            ron.throw([harry.x, harry.y])
        ron.throw()
    elif len(the_list_of_snaffles) > 1:
        ron.move_to_snaffle()
    else:
        print(f"Ron protecting Rings", file=sys.stderr, flush=True)
        ron.protect_rings()

    if snaffle := snaffle_close_to_goal():
        if snaffle.velocity_to_goal() < 10:
            if my_magic >= FLIPENDO_COST:
                my_magic -= FLIPENDO_COST
                ron.flipendo(snaffle)

    if my_magic > 30:
        harry.petrificus(draco)


    print(harry.do_the_thing)
    print(ron.do_the_thing)