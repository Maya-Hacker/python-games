




import random


def askForGuess():
    while True:
        guess = input('> ') # Enter the guess

        if guess.isdecimal():
            return int(guess)
        print('Please eter a number between a and 100.')


print('Guess the Number!')
print()
secretNumber = random.randint(1, 100) # Select random number.
print('I am thinking of  number between 1 and 100.')

for i in range(10): # Give player max 10 guesses.
    print('You have {} guesses left. Take a guess.'.format(10 -i))

    guess = askForGuess()
    if guess == secretNumber:
            break

    # OFfer hint
    if guess < secretNumber:
        print('Your guess is too low.')
    if guess > secretNumber:
        print('Your guess is too high.')

# Reveal the results
if guess == secretNumber:
    print('YAY!!! You guessed my number!')
else:
    print('GAME OVER. The number I was thinking of was', secretNumber)