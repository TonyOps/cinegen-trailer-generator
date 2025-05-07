import json
import numpy as np
import random
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def scene_to_string(scene):
    dur = scene['duracao_segundos']
    return f"{scene['elemento']} ({scene['contexto']} - {dur:.2f}s) - NV: {scene['NV']}"

with open('banco.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

trailers = []
for trailer in data['trailers']:
    cenas = trailer['cenas']
    scene_list = [scene_to_string(scene) for scene in cenas]
    trailers.append(scene_list)

delimiter = " <SCENE> "
texts = [delimiter.join(scene_list) for scene_list in trailers]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = np.eye(total_words)[y]

model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X, y, epochs=200, verbose=1)

def generate_trailer(num_scenes):
    seed_text = random.choice(texts)
    generated_text = seed_text
    while generated_text.count("<SCENE>") < (num_scenes - 1):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        generated_text += " " + output_word

    scenes_generated = [s.strip() for s in generated_text.split("<SCENE>") if s.strip() != ""]
    scenes_generated = scenes_generated[:num_scenes]

    timestamp = 0.0  
    output_lines = []
    order = 1
    for scene_str in scenes_generated:
        match = re.search(r'\((\w)\s*-\s*([\d\.]+)s\)', scene_str)
        if match:
            contexto = match.group(1)
            dur = float(match.group(2))
        else:
            contexto = "L"
            dur = 3.0  

        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        frames = int((timestamp - int(timestamp)) * 100)
        ts_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

        dur_str = f"{dur:.2f}s"
        sec_part = int(dur)
        ms_part = int(round((dur - sec_part) * 1000))
        detailed = f"[{sec_part}s {ms_part}ms]"

        nv_match = re.search(r'NV:\s*(NV\d+)', scene_str)
        nv = nv_match.group(1) if nv_match else "NV1"

        line = f"{ts_str} - {scene_str} - Ordem: {order} - NV: {nv}"
        output_lines.append(line)
        order += 1
        timestamp += dur  

    return output_lines

if __name__ == '__main__':
    try:
        num_scenes = int(input("Digite o número de cenas desejado: "))
    except ValueError:
        print("Por favor, digite um número inteiro válido.")
        exit(1)

    generated_trailer = generate_trailer(num_scenes)
    print("\nTrailer gerado:")
    for line in generated_trailer:
        print(line)
    
    save_option = input("\nDeseja salvar o resultado em um arquivo? (s/n): ").strip().lower()
    if save_option == "s":
        file_name = input("Digite o nome do arquivo (ex: trailer.txt): ").strip()
        if not file_name:
            file_name = "trailer.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            for line in generated_trailer:
                f.write(line + "\n")
        print(f"Resultado salvo no arquivo '{file_name}'.")
    
    input("\nPressione Enter para sair...")
