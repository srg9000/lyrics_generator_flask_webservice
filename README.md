# lyrics_generator_flask_webservice
This is a flask app that takes in a string from user, converts it to a seed to generate a random string of lyrics by using an LSTM model.
The LSTM network is currently trained with only a single LSTM and single dense layer for approximately 50 iterations due to lack of computational resources.
The model has been trained on lyrics from only one artist (stage name: Logic) as previous attempts with a larger dataset were computationally expensive.

Relevant paper: https://www.aclweb.org/anthology/D15-1221/
