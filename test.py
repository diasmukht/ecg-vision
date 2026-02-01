
word = 'python'

secret_word = '_' * len(word)
# secret_word = ['_'] * len(word)



while True:

    letter = input('введите букву: ').lower()
    
    while len(letter) != 1:
        print('---Ошибка! Введите ровно одну букву (пустой ввод запрещен).---')
        letter = input('введите букву: ').lower()

    index = 0
    
    while index < len(word):
        if letter == word[index]:
            secret_word = secret_word[:index] + letter + secret_word[index + 1:]
            # secret_word[index] = letter
        index += 1

    if secret_word == word:
        print('---Вы прошли игру!---')
        break

    print(secret_word)


