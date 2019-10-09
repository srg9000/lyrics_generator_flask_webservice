from flask import Flask,render_template,url_for,request

import pickle

import numpy
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


def train():
		# Small LSTM Network to Generate Text for Alice in Wonderland

	# load ascii text and covert to lowercase
	filename = "lyrics.txt"
	raw_text = open(filename, 'r', encoding='utf-8').read()
	raw_text = raw_text.lower()
	# create mapping of unique chars to integers
	chars = raw_text.split()
	char_to_int = dict((c,i) for i,c in enumerate(chars))
	# summarize the loaded data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print ("Total Characters: ", n_chars)
	print ("Total Vocab: ", n_vocab)
	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print ("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='relu'))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# define the checkpoint
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	# fit the model
	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = int.from_bytes(bytes(message,'utf-8'),"big")

#	model = Sequential()
#	model.add(LSTM(256, input_shape=(100, 1)))
#	model.add(Dropout(0.2))
#	model.add(Dense(51, activation='relu'))
#	model.add(Dense(51, activation='softmax'))
#	model.compile(loss='categorical_crossentropy', optimizer='adam')
#	model.load_weights("model.hdf5")
	model = load_model("model_1.7654.h5")
	chars = ['\n', ' ', '!', '"', '&', "'", '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

	int_to_char = dict((i, c) for i, c in enumerate(chars))
	f = open("dataX.pkl",'rb')
	dataX = pickle.load(f)
	# pick a random seed
	start = data%len(dataX)#numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
#	print "Seed:"
#	print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

	result_arr = []
	# generate characters
	for i in range(1000):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(51)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]

		result_arr.append(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]


		print(''.join(result_arr))
#		vect = cv.transform(data).toarray()
#		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = ''.join(result_arr))



if __name__ == '__main__':
#	app.debug = True
	app.run(debug = False)
