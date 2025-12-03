from random import randint
from time import time 


while  True:
    start = int(input("Введите начало диапазона- "))
    end = int(input("Введите конец диапазона- "))
    z = 0
    num_sik = randint(start, end)
    start_time = time()
    game = True
    while game == True:
        z += 1
        num_ans = int(input("Попрорбуйте угадать число: "))
        if num_ans == num_sik:
            end_time = time ()
            print(f"Вы отгадали!!! Количество попыток - {z}! Затраченное время - {int(end_time - start_time)} секунд")
            break
        elif num_ans > num_sik:
            print("Загаднаное число меньше")
        elif num_ans < num_sik:
            print("Загаданное число больше")
    retern_game = input("Хотите сыграть еще раз, Да/Нет?: ")
    if retern_game == "нет":
        game = False
        break