# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_network.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/11 16:55:26 by aptive            #+#    #+#              #
#    Updated: 2022/11/13 20:33:20 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from	fonction_neural import *
from	sklearn.datasets import make_blobs, make_circles, make_moons
from	tqdm import tqdm


def neural_network(X, y, n1=32, learning_rate = 0.1, n_iter = 1000):

	# initialisation parametres
	n0 = X.shape[0]
	n2 = y.shape[0]
	np.random.seed(0)
	parametres = initialisation(n0, n1, n2)

	train_loss = []
	train_acc = []
	history = []

	# gradient descent
	for i in tqdm(range(n_iter)):
		activations = forward_propagation(X, parametres)
		A2 = activations['A2']

		# Plot courbe d'apprentissage
		train_loss.append(log_loss(y.flatten(), A2.flatten()))
		y_pred = predict(X, parametres)
		train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

		history.append([parametres.copy(), train_loss, train_acc, i])

		# mise a jour
		gradients = back_propagation(X, y, parametres, activations)
		parametres = update(gradients, parametres, learning_rate)


	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.plot(train_loss, label='train loss')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(train_acc, label='train acc')
	plt.legend()
	plt.show()

	return parametres


# main


# print(predict(new_plant, W, b))

# Generation du dataset ---------------------------------------------------------

# dataset
# X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2)

# X, y = make_circles(n_samples=100, noise=0.40, factor=0.5, random_state=10)
X, y = make_moons(n_samples=100, noise=.05)

X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

# plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')


# plt.show()


parametres = neural_network(X, y, n1=16, n_iter= 10000, learning_rate=0.1)

# print(parametre)

# W2 = parametre['W2']
# b2 = parametre['b2']


# print("W2: " , W2[0][0])
# print("W2: " , W2[0][1])

# print("W2 [0]: " , b2[0][0])


# x0 = np.linspace(-1, 4, 100)
# x1 = (-W2[0][0] * x0 - b2[0][0]) / W2[0][1]


# plt.figure(figsize=(6, 4))
# plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
# plt.plot(x0, x1, c='orange', lw=3)
# plt.show()


fig, ax = plt.subplots()
ax.scatter(X[0, :], X[1, :], c=y, cmap='summer')
x0_lim = ax.get_xlim()
x1_lim = ax.get_ylim()


resolution = 100
x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)

# mshgrid

X0, X1 = np.meshgrid(x0, x1)
print(X0.shape)
print(X0[:4, :4])

#assemble (100, 100) -> 10000, 2

XX = np.vstack((X0.ravel(), X1.ravel()))


print(XX.shape)

Z = predict(XX, parametres)
print(Z.shape)

Z = Z.reshape((resolution, resolution))

ax.pcolormesh(X0, X1, Z, cmap='summer', alpha=0.3, zorder =-1)
ax.contour(X0, X1, Z, color='orange')

plt.show()
