import numpy as np
import msvcrt
from hopfield import test_incremental_hopfield_network
from hamming import test_incremental_hamming_network
from bidirectional_associative_memory import test_incremental_bam


def main_menu():
    while True:
        print("\n=== Главное меню ===")
        print("1. Тестировать нейронную сеть Хэмминга")
        print("2. Тестировать нейронную сеть Хопфилда")
        print("3. Тестировать двунаправленную ассоциативную память")
        print("q. Выход")
        print("Выберите действие (нажмите соответствующую клавишу без Enter): ", end="", flush=True)
        key = msvcrt.getch().decode('utf-8')
        print(key)
        if key == '1':
            print("\nЗапуск тестирования сети Хэмминга...")
            print("*" * 100, "\n")
            test_incremental_hamming_network()
            print("*" * 100, "\n")
        elif key == '2':
            print("Выберите режим работы НС Хопфилда:")
            print("1. Асинхронный")
            print("2. Синхронный")
            print("Нажмите клавишу (без Enter): ", end="", flush=True)
            mode = msvcrt.getch().decode('utf-8')
            print(mode)
            if mode not in ['1', '2']:
                print("Недопустимый выбор! По умолчанию выбран режим 1.")
                mode = '1'
            print("\nЗапуск тестирования сети Хопфилда...")
            print("*"*100)
            test_incremental_hopfield_network(mode)
            print("*" * 100, "\n")
        elif key.lower() == '3':
            print("\nЗапуск тестирования двунаправленной ассоциативной памяти...")
            print("*" * 100, "\n")
            test_incremental_bam()
            print("*" * 100, "\n")
        elif key.lower() == 'q':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Пожалуйста, повторите ввод.")


if __name__ == "__main__":
    main_menu()
