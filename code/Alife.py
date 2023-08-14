import math
import random
import copy
import pygame
pygame.init()

WIDTH = 500
HEIGHT = 500
FPS = 30

def quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.event.clear()
            return True
    return False

def clear(screen):
    screen.fill((100, 100, 100))

def update(fps):
    pygame.display.update()
    pygame.time.Clock().tick(fps)

def ngon(screen, color = (255, 255, 255), gx = 100, gy = 100, gr = 100, n = 5, rot = 0):
    points = [(gr*math.cos(th) + gx, gr*math.sin(th) + gy) for th in [i*2*math.pi/n - math.pi/2 + rot for i in range(n)]]
    pygame.draw.polygon(screen, color=color, points=points, width=1)

XMIN, XMAX = -1.5, 1.5
YMIN, YMAX = -1.5, 1.5

class Entity:

    def __init__(self, hue=0, sat=0, brightness=1, radius=0.1):
        self.hue = hue # [0, 1]
        self.sat = sat # [0, 1]
        self.brightness = brightness # [0, 1]
        self._color = pygame.Color(0)
        self._updateColor()
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.radius = radius
        self.alive = True

    def draw(self, screen, color, gx, gy, gr):
        ngon(screen, color, gx, gy, gr, 4)

    def _updateColor(self):
        self._color.hsva = 360*self.hue, 100*self.sat, 100*self.brightness, 100

    def update(self, screen):
        if self.alive:
            self._updateColor()
            # Processing �� map �֐����֗��Ȃ̂Ŏ������R�s�[
            def map(value, istart, istop, ostart, ostop):
                return ostart + (ostop - ostart)*((value - istart)/(istop - istart))
            gx = map(self.x, XMIN, XMAX, 0, WIDTH)
            gy = map(self.y, YMIN, YMAX, HEIGHT, 0)
            gr = map(self.radius, 0, XMAX, 0, WIDTH/2)
            self.draw(screen, self._color, gx, gy, gr)

    def collideWith(self, other) -> bool:
        dx = self.x - other.x
        dy = self.y - other.y
        rr = self.radius + other.radius
        return dx*dx + dy*dy < rr*rr

class C(Entity):

    def __init__(self, hue, omega):
        super().__init__(hue=hue, sat=1)
        self._theta = 0
        self.omega = omega

    def update(self, screen):
        self.x = math.cos(self._theta)
        self.y = math.sin(self._theta)
        self._theta += self.omega
        super().update(screen)

class B(Entity):

    def __init__(self, hue, omega):
        super().__init__(hue=hue, sat=1)
        self._theta = 0
        self.omega = omega

    def update(self, screen):
        self.radius = 1.0*math.fabs(math.cos(self._theta))
        self._theta += self.omega
        super().update(screen)

class EntityWithLife(Entity):

    def __init__(self, hue=0, sat=0, life=100):
        self._life0 = life
        self._rot = random.uniform(0, 2*math.pi)
        self.reset()
        super().__init__(hue=hue, sat=sat)

    def update(self, screen):
        if 0 < self.life:
            self.life -= 1
            self._age += 1
        else:
            self.life = 0;
            self.alive = False

        self._rot += min(0.3, 0.005*self.life)
        self.radius = min(0.1, 0.001*self.life)

        super().update(screen)

    def draw(self, screen, color, gx, gy, gr):
        ngon(screen, color, gx, gy, gr, 3, self._rot)

    def reset(self):
        self.life = self._life0
        self.alive = True
        self._age = 0

    def isReproductive(self):
        return 30 < self._age and 10 < self.life

class RandomWalker(EntityWithLife):

    speed = 0.01

    def __init__(self, hue=0, sat=0, life=100):
        th = random.random()*2*math.pi
        self._vx = RandomWalker.speed*math.cos(th)
        self._vy = RandomWalker.speed*math.sin(th)
        super().__init__(hue=hue, sat=sat, life=life)

    def update(self, screen):
        d = 0.5
        th = math.atan2(self._vy, self._vx) + random.uniform(-d, d)
        self._vx = RandomWalker.speed*math.cos(th)
        self._vy = RandomWalker.speed*math.sin(th)
        self.x += self._vx
        self.y += self._vy

        if self.x < -1 or 1 < self.x:
            self._vx *= -1
        if self.y < -1 or 1 < self.y:
            self._vy *= -1

        super().update(screen)

class Food(Entity):

    def __init__(self, hue=0.2):
        super().__init__(hue=hue, sat=1, radius=0.05)

class RandomWalkerA(RandomWalker):

    def __init__(self, life=100, reproductionRate=0.4):
        self.reproductionRate = reproductionRate
        super().__init__(hue=0, sat=1, life=life)

class RandomWalkerB(RandomWalker):

    def __init__(self, life=105, reproductionRate=0.35):
        self.reproductionRate = reproductionRate
        super().__init__(hue=0.3, sat=1, life=life)

def drawWithSex(self, screen, color, gx, gy, gr):
    n = [ 3, 4, 5 ]
    ngon(screen, color, gx, gy, gr, n[self.sex], self._rot)

class RandomWalkerAWithSex(RandomWalkerA):
    def __init__(self, life=100, reproductionRate=0.4):
        self.sex = random.randint(0, 2)
        super().__init__(life=life, reproductionRate=reproductionRate)
RandomWalkerAWithSex.draw = drawWithSex

class RandomWalkerBWithSex(RandomWalkerB):
    def __init__(self, life=105, reproductionRate=0.35):
        self.sex = random.randint(0, 2)
        super().__init__(life=life, reproductionRate=reproductionRate)
RandomWalkerBWithSex.draw = drawWithSex

entitiesA = [RandomWalkerAWithSex() for _ in range(50)]
entitiesB = [RandomWalkerBWithSex() for _ in range(50)]
entities = entitiesA + entitiesB
MAX_ENTITIES = len(entities)

foods = [Food() for _ in range(100)]
MAX_FOODS = len(foods)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

while not quit():
    clear(screen)
    random.shuffle(entities)
    numEntities = len(entities)
    babies = []

    for e in entities:

        for f in [f for f in foods if e.collideWith(f)]:
            e.life += 5
            foods.remove(f)
            break

        for o in [o for o in entities if o is not e and e.collideWith(o)]:
            if type(o) is not type(e):
                # バトル
                e.life -= 5
                o.life -= 5
            elif e.sex != o.sex and e.isReproductive() and o.isReproductive():
                # 有性生殖
                be = random.sample([e, o], 1)[0]
                if numEntities < MAX_ENTITIES and 1 - be.reproductionRate < random.random():
                    baby = copy.copy(be)
                    baby.reset()
                    e.life -= 5
                    o.life -= 5
                    babies.append(baby)
                    numEntities += 1

    foods += [Food() for _ in range(MAX_FOODS - len(foods))]
    for e in foods + entities:
        e.update(screen)

    entities = babies + [x for x in entities if x.alive]

    update(FPS)

pygame.display.quit()









