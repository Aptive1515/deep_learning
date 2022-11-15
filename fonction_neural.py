# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fonction_neural.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/13 17:35:46 by aptive            #+#    #+#              #
#    Updated: 2022/11/14 13:38:08 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import	numpy as np
import	matplotlib.pyplot as plt
from	sklearn.metrics import accuracy_score, log_loss

# Fonction initialisation -------------------------------------------------------
def initialisation(dimension):

	parametres = {}
	C = len(dimension)

	for c in range(1, C):
		parametres['W' + str(c)] = np.random.randn(dimension[c], dimension[c - 1])
		parametres['b' + str(c)] = np.zeros((dimension[c], 1))

	return parametres


# Fonction Modele/ Forward propagation ------------------------------------------
def forward_propagation(X, parametres):

	activations = {'A0' : X}
	C = len(parametres) // 2

	for c in range(1, C + 1):
		Z = parametres['W' + str(c)].dot(activations['A' + str(c -1)]) + parametres['b' + str(c)]
		activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

	return activations

# Fonction LOGLOSS -------------------------------------------------------------
# def log_loss( A, y):
# 	return 1 / len(y) * np.sum( -y * np.log(A) - (1 - y) * np.log(1 - A))

# Fonction Gradient / BACKPROPAGATION -------------------------------------------
def back_propagation(y,activations, parametres):

	m = y.shape[1]
	C = len(parametres) // 2

	dZ = activations['A' + str(C)] - y
	gradients = {}

	for c in reversed(range(1, C + 1)):
		gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
		gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims = True)
		if c > 1:
			dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

	return gradients

# Fonction UPDATE ---------------------------------------------------------------
def	update(gradients, parametres, learning_rate):

	C = len(parametres) // 2

	for c in range(1, C + 1):
		parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
		parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

	return parametres

# predict -----------------------------------------------------------------------
def predict(X, parametres):

	activations = forward_propagation(X, parametres)

	C = len(parametres) // 2
	af = activations['A' + str(C)]
	# print('probabilite : ', A2)
	return af >= 0.5
