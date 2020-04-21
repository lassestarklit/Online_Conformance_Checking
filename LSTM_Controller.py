import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


class LSTMController:

    def one_hot_encode(self, input_activity):
        """

        :param input_activity: activity label to one-hot-encode
        :return:
        """
        labels = ['A', 'H', 'J', 'C', 'I', 'K', 'E', 'F', 'G', 'D', 'B']

        labels = ['Activity ' + activity for activity in labels]

        one_hot_encoded = [1 if activity == input_activity else 0 for activity in labels]

        return one_hot_encoded

    def prepare_training_data(self, event):
        """
        :param data: case from XES log
        :return:
        """

        activity_label = event["concept:name"]
        legal_bool = event["concept:legal"]
        X_train_list = self.one_hot_encode(activity_label)
        self.X_train = numpy.array(X_train_list)
        self.X_train = self.X_train.reshape((1, 1, 11))

        self.y_train = [1 if legal_bool else 0]

    def create_model(self, memory_units=100, loss_function='binary_crossentropy', optimizer='adam'):
        """

        :param batch: Batch of input
        :return:
        """
        # create the model
        self.model = Sequential()
        '''self.model.add(LSTM(memory_units))
        self.model.add(Dense(1,input_shape=(11,),activation='sigmoid')) # Since classification problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1

        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])'''
        # For a single-input model with 2 classes (binary classification):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(1, 11)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit_model(self, epochs=3, batch_size=1):
        # Batch size 1 (Stochastic Gradient Descent) since stream data
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
        print("Fitted")


