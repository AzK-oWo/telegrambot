# telegrambot

## what it is about

The telegrambots, which one of them can collect voice data of digit and another can predict the digit you called

Bot has trainig model which basic on dataset before predict your numbers

## how to run the dataset bot

	make bot_dataset

then send the voice with 5 digits to [bot](https://t.me/Listen_to_your_numbers_oWo_bot)

## how to split your digits from voice data

	make vad_sort
	make splitted

## how to train your model

	make training

## how to run recognition bot

	make bot_recognition

then send the voice with 1 digit to [bot](https://t.me/Say_digit_get_number_oWo_bot) and get prediction digit
