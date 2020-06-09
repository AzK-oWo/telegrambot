import pickle

filename = "model.pkl"
with open(filename, 'rb') as f:
    model_pickled = f.read()
model = pickle.loads(model_pickled)

file_path = "dataset/inference/unk.wav"
sample_rate, audio = read(file_path)

max_duration_sec = 0.6
max_duration = int(max_duration_sec * sample_rate + 1e-6)
if len(audio) < max_duration:
    audio = np.pad(audio, (0, max_duration - len(audio)), constant_values=0)
feature = librosa.feature.melspectrogram(audio.astype(float), sample_rate, n_mels=16, fmax=1000)
features_flatten = feature.reshape(-1)

answer = model.predict([features_flatten])[0]
print(answer)