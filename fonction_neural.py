# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fonction_neural.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/13 17:35:46 by aptive            #+#    #+#              #
#    Updated: 2022/11/13 18:28:18 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import	numpy as np
import	matplotlib.pyplot as plt
from	sklearn.metrics import accuracy_score, log_loss

# Fonction initialisation -------------------------------------------------------
def initialisation(n0, n1, n2):
	W1 = np.random.randn(n1, n0)
	b1 = np.zeros((n1, 1))
	W2 = np.random.randn(n2, n1)
	b2 = np.zeros((n2, 1))

	parametres = {
		'W1' : W1,
		'b1' : b1,
		'W2' : W2,
		'b2' : b2
	}

	return parametres


# Fonction Modele/ Forward propagation ------------------------------------------
def forward_propagation(X, parametres):

	W1 = parametres['W1']
	b1 = parametres['b1']
	W2 = parametres['W2']
	b2 = parametres['b2']

	Z1 = W1.dot(X) + b1
	A1 = 1 / (1 + np.exp(-Z1))

	Z2 = W2.dot(A1) + b2
	A2 = 1 / (1 + np.exp(-Z2))

	activations = {
		'A1': A1,
		'A2': A2
	}

	return activations

# Fonction LOGLOSS -------------------------------------------------------------
# def log_loss( A, y):
# 	return 1 / len(y) * np.sum( -y * np.log(A) - (1 - y) * np.log(1 - A))

# Fonction Gradient / BACKPROPAGATION -------------------------------------------
def back_propagation(X, y, parametres, activations):

	A1 = activations['A1']
	A2 = activations['A2']
	W2 = parametres['W2']

	m = y.shape[1]

	dZ2 = A2 - y
	dW2 = 1 / m * dZ2.dot(A1.T)
	db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

	dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
	dW1 = 1 / m * dZ1.dot(X.T)
	db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

	gradients = {
		'dW1' : dW1,
		'db1' : db1,
		'dW2' : dW2,
		'db2' : db2
	}

	return gradients

# Fonction UPDATE ---------------------------------------------------------------
def	update(gradients, parametres, learning_rate):

	W1 = parametres['W1']
	b1 = parametres['b1']
	W2 = parametres['W2']
	b2 = parametres['b2']

	dW1 = gradients['dW1']
	db1 = gradients['db1']
	dW2 = gradients['dW2']
	db2 = gradients['db2']

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parametres = {
		'W1' : W1,
		'b1' : b1,
		'W2' : W2,
		'b2' : b2
	}

	return parametres

# predict -----------------------------------------------------------------------
def predict(X, parametres):

	activations = forward_propagation(X, parametres)

	A2 = activations['A2']
	# print('probabilite : ', A2)
	return A2 >= 0.5
