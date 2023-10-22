import random
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP
import sys

FPS = 32
scr_width = 289
scr_height = 511
display_screen_window = pygame.display.set_mode((scr_width, scr_height))
play_ground = scr_height * 0.8
game_image = {}
game_audio_sound = {}


def welcome_main_screen():
    # Affiche l'ecran de bienvenue du jeu
    p_x = int(scr_width / 5)
    p_y = int((scr_height - game_image['joueur'].get_height()) / 2)
    msgx = int((scr_width - game_image['message'].get_width()) / 2)
    msgy = int(scr_height * 0.13)
    b_x = 0
    while True:
        for event in pygame.event.get():
            # si lutilisateur clique sur le bouton croix, quitte le jeu
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

            # Si le joueur appuie sur espace ou sur la touche haut, lance le jeu
            elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                return
            else:
                display_screen_window.blit(game_image['fond'], (0, 0))
                display_screen_window.blit(game_image['joueur'], (p_x, p_y))
                display_screen_window.blit(game_image['message'], (msgx, msgy))
                display_screen_window.blit(game_image['base'], (b_x, play_ground))
                pygame.display.update()
                time_clock.tick(FPS)


def main_gameplay():
    score = 0
    p_x = int(scr_width / 5)
    p_y = int(scr_width / 2)
    b_x = 0

    n_pip1 = get_Random_Tuyaux()
    n_pip2 = get_Random_Tuyaux()

    # Calcul de la taille de tuyaux qui seront affiches a l'ecran
    up_pips = [
        {'x': scr_width + 200, 'y': n_pip1[0]['y']},
        {'x': scr_width + 200 + (scr_width / 2), 'y': n_pip2[0]['y']},
    ]

    low_pips = [
        {'x': scr_width + 200, 'y': n_pip1[1]['y']},
        {'x': scr_width + 200 + (scr_width / 2), 'y': n_pip2[1]['y']},
    ]

    pip_Vx = -4

    p_vx = -9
    p_mvx = 10
    p_mvy = -8
    p_accuracy = 1

    p_flap_accuracy = -8
    p_flap = False

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if p_y >= 0:
                    p_vx = p_flap_accuracy
                    p_flap = True
                    game_audio_sound['wing'].play()

        # Test de collision
        cr_tst = is_Colliding(p_x, p_y, up_pips, low_pips)
        if cr_tst:
            return

        p_middle_positions = p_x + game_image['joueur'].get_width() / 2
        for tuyau in up_pips:
            pip_middle_positions = tuyau['x'] + game_image['tuyau'][0].get_width() / 2
            if pip_middle_positions <= p_middle_positions <= pip_middle_positions + 4:
                score += 1
                print(f"Votre score est de {score}")
                game_audio_sound['point'].play()

        if p_vx <= p_mvx and not p_flap:
            p_vx += p_accuracy

        if p_flap:
            p_flap = False
        p_height = game_image['joueur'].get_height()
        p_y = p_y + min(p_vx, play_ground - p_y - p_height)

        for pip_upper, pip_lower in zip(up_pips, low_pips):
            pip_upper['x'] += pip_Vx
            pip_lower['x'] += pip_Vx

        if 0 <= up_pips[0]['x'] <= 5:
            new_pip = get_Random_Tuyaux()
            up_pips.append(new_pip[0])
            low_pips.append(new_pip[1])
        if up_pips[0]['x'] <= -game_image['tuyau'][0].get_width():
            up_pips.pop(0)
            low_pips.pop(0)

        # Affichage du fond d'ecran et des tuyaux
        display_screen_window.blit(game_image['fond'], (0, 0))
        for pip_upper, pip_lower in zip(up_pips, low_pips):
            display_screen_window.blit(
                game_image['tuyau'][0], (pip_upper['x'], pip_upper['y'])
            )
            display_screen_window.blit(
                game_image['tuyau'][1], (pip_lower['x'], pip_lower['y'])
            )

        # Affichage du joueur
        display_screen_window.blit(game_image['base'], (b_x, play_ground))
        display_screen_window.blit(game_image['joueur'], (p_x, p_y))
        d = [int(x) for x in list(str(score))]
        w = 0
        for digit in d:
            w += game_image['numbers'][digit].get_width()
            Xoffset = (scr_width - w) / 2

        # Affichage du nombre de tuyaux passes a l'ecran
        for digit in d:
            display_screen_window.blit(
                game_image['numbers'][digit], (Xoffset, scr_height * 0.12)
            )
            Xoffset += game_image['numbers'][digit].get_width()

        pygame.display.update()
        time_clock.tick(FPS)


def is_Colliding(p_x, p_y, up_tuyaux, low_tuyaux):
    if p_y >= play_ground - 25 or p_y <= 0:
        game_audio_sound['hit'].play()
        return True
    
    for tuyau in up_tuyaux:
        pip_h = game_image['tuyau'][0].get_height()
        if (p_y <= pip_h + tuyau['y'] and abs(p_x - tuyau['x']) <= game_image['tuyau'][0].get_width()//2):
            game_audio_sound['hit'].play()
            return True
        
    for tuyau in low_tuyaux:
        if (p_y + game_image['joueur'].get_height() >= tuyau['y']) and abs(p_x - tuyau['x']) <= game_image['tuyau'][0].get_width()//2:
            game_audio_sound['hit'].play()
            return True
        
        return False

def get_Random_Tuyaux():
    pip_h = game_image['tuyau'][0].get_height()
    off_s = scr_height / 3
    yes2 = off_s + random.randrange(
        0, int(scr_height - game_image['base'].get_height() - 1.2 * off_s)
    )
    tuyauX = scr_width + 10
    y1 = pip_h - yes2 + off_s
    tuyau = [
        {'x': tuyauX, 'y': -y1},  # upper Tuyau
        {'x': tuyauX, 'y': yes2},  # lower Tuyau
    ]
    return tuyau


if __name__ == "__main__":
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pygame.init()
    time_clock = pygame.time.Clock()
    joueur = this_dir+'/images/bird.png'
    bcg_image = this_dir+'/images/background.png'
    tuyau_image = this_dir+'/images/pipe.png'
    sound_extension = 'ogg'

    pygame.display.set_caption('Flappy bird Game')
    game_image['numbers'] = (
        pygame.image.load(this_dir+'/images/0.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/1.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/2.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/3.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/4.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/5.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/6.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/7.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/8.png').convert_alpha(),
        pygame.image.load(this_dir+'/images/9.png').convert_alpha(),
    )

    game_image['message'] = pygame.image.load(this_dir+'/images/message.png').convert_alpha()
    game_image['base'] = pygame.image.load(this_dir+'/images/base.png').convert_alpha()
    game_image['tuyau'] = (
        pygame.transform.rotate(pygame.image.load(tuyau_image).convert_alpha(), 180),
        pygame.image.load(tuyau_image).convert_alpha(),
    )

    # Sons du jeu
    game_audio_sound['die'] = pygame.mixer.Sound(this_dir+'/sounds/die.'+sound_extension)
    game_audio_sound['hit'] = pygame.mixer.Sound(this_dir+'/sounds/hit.'+sound_extension)
    game_audio_sound['point'] = pygame.mixer.Sound(this_dir+'/sounds/point.'+sound_extension)
    game_audio_sound['swoosh'] = pygame.mixer.Sound(this_dir+'/sounds/swoosh.'+sound_extension)
    game_audio_sound['wing'] = pygame.mixer.Sound(this_dir+'/sounds/wing.'+sound_extension)

    game_image['fond'] = pygame.image.load(bcg_image).convert()
    game_image['joueur'] = pygame.image.load(joueur).convert_alpha()

    while True:
        welcome_main_screen()
        main_gameplay()