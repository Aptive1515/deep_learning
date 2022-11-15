# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    visualization.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/14 14:04:55 by aptive            #+#    #+#              #
#    Updated: 2022/11/14 18:05:32 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from neural_network import *


def visualization(X, y, parametres, train_loss, train_accu, iter):




	ax = np.array([], [])


	# ax[0].clear() #frontiere de decision
	# ax[1].clear() #sigmoide
	# ax[2].clear() #fonction cout

	s = 300
	resolution = 100


	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40,10))

	#frontiere de decision
	# ax[0].scatter(0, 10, c=y,s=s, cmap ="summer", edgecolor="k", linewidths=1)


	# ax[0].set_xlim(0, 10)
	# ax[0].set_ylim(0, 10)

	ax[0].set_title('Frontiere de Decision')
	ax[0].set_xlabel('x1')
	ax[0].set_ylabel('x2')
	ax[0].scatter(X[0, :], X[1, :], c=y, cmap='summer')

	x0_lim = ax[0].get_xlim()
	x1_lim = ax[0].get_ylim()
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

	ax[0].pcolormesh(X0, X1, Z, cmap='summer', alpha=0.3, zorder =-1)
	ax[0].contour(X0, X1, Z)

	# Graph 2 -------------------------------------------------------------------------

	ax[1].set_title('train loss')
	ax[1].set_xlabel('Z')
	ax[1].set_ylabel('A(Z)')

	ax[1].plot(train_loss, c='red', lw=1)

	# ax[1].set_xlim(0, iter)
	# ax[1].set_ylim(train_loss[iter], train_loss[0])

	ax[1].set_xlim(0, iter)
	ax[1].set_ylim(0, train_loss[0] * 1.1)
	ax[1].set_title('Fonction Cout')
	ax[1].set_xlabel('iteration = \n' + 'Vraisemblence = ')
	ax[1].set_ylabel('Loss')


	# plt.plot(train_loss, label='train loss')


	# Graph 3 -------------------------------------------------------------------------

	ax[2].set_xlim(0, iter)
	ax[2].set_ylim(0, 1.1)
	ax[2].set_title('Fonction Cout')
	ax[2].set_xlabel('iteration')
	ax[2].set_ylabel('Vraisemblence')
	ax[2].plot(train_accu, c='red', lw=1)

	# plt.show()
