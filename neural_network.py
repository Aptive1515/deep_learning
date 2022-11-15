# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_network.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/11 16:55:26 by aptive            #+#    #+#              #
#    Updated: 2022/11/14 19:02:21 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from	fonction_neural import *
from	generator_datasets import *
from	tqdm import tqdm


def neural_network(X, y, hidden_layers = (32, 32, 32), learning_rate = 0.1, n_iter = 1000):

	# initialisation parametres
	np.random.seed(0)
	dimensions = list(hidden_layers)
	dimensions.insert(0, X.shape[0])
	dimensions.append(y.shape[0])
	parametres = initialisation(dimensions)

	train_loss = []
	train_acc = []
	history = []

	# gradient descent
	for i in tqdm(range(n_iter)):
		activations = forward_propagation(X, parametres)
		gradients = back_propagation(y, activations, parametres)
		parametres = update(gradients, parametres, learning_rate)

		# A2 = activations['A2']

		# Plot courbe d'apprentissage
		C = len(parametres) // 2

		train_loss.append(log_loss(y.flatten(), activations['A' + str(C)].flatten()))
		y_pred = predict(X, parametres)
		train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
		history.append([parametres.copy(), train_loss, train_acc, i])

	return history


# main


# print(predict(new_plant, W, b))

# Generation du dataset ---------------------------------------------------------





