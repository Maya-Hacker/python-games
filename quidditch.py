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
        return f"({self.x}, {self.y})"

    def dist(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

# example
goal_pos_R = Position(16000, 3750)
goal_pos_L = Position(0, 3750)
goal_pos_R.x
print(f"{goal_pos_R=}, {goal_pos_R.dist(goal_pos_L)}", file=sys.stderr, flush=True)
# fff

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

    def throw(self, x_y = None):
        if my_team_id == 0:
            goal_x = 16000
        else:
            goal_x = 0
        goal_y = 3750
        if x_y is not None:
            goal_x, goal_y = x_y
        self.do_the_thing = f"THROW {goal_x} {goal_y} 500"

    def move_to_snaffle(self, snaffleindex: int):
        # print(f"move_to_snaffle {snaffleindex=}", file=sys.stderr, flush=True)
        the_snaffle = the_list_of_snaffles[snaffleindex]
        self.do_the_thing = f"MOVE {the_snaffle.x} {the_snaffle.y} {self.thrust}"

    def protect_rings(self):
        if my_team_id == 0:
            goal_x = 3000
        else:
            goal_x = 13000
        goal_y = 3750
        self.do_the_thing = f"MOVE {goal_x} {goal_y} {self.thrust}"


    
the_list_of_snaffles: list[Snaffle] = []


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

        x = int(inputs[2])  # position
        y = int(inputs[3])  # position
        vx = int(inputs[4])  # velocity
        vy = int(inputs[5])  # velocity
        state = int(inputs[6])  # 1 if the wizard is holding a Snaffle, 0 otherwise

    if harry.has_snaffle:
        harry.throw()
    else:
        harry.move_to_snaffle(0)

    if ron.has_snaffle:
        # ron.throw([harry.x, harry.y])
        ron.throw()
    elif len(the_list_of_snaffles) > 1:
        ron.move_to_snaffle(1)
    else:
        ron.protect_rings()



    print(harry.do_the_thing)
    print(ron.do_the_thing)