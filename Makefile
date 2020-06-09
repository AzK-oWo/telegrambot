bot_dataset:
	python3 audio_digits_dataset_bot.py
bot_recognition:
	python3 audio_digits_recognition_bot.py	
vad_sort:
	{ \
	for fname in $$(ls dataset/wav/* | grep wav); \
	do \
		python3 split_by_vad.py $$fname 0.1 0.005 dataset/splitted; \
	done \
	} 
splitted:
	cp dataset/splitted/* dataset/splitted_final/ -r 
	rm dataset/splitted/*/*
	rm dataset/wav/* 
	rm dataset/ogg/* 
training:
	python3 training.py