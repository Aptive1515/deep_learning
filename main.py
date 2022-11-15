# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/14 13:59:36 by aptive            #+#    #+#              #
#    Updated: 2022/11/14 20:43:16 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from neural_network import *
from visualization import *
from matplotlib.animation import FuncAnimation


plt.style.use('dark_background')

X, y = generator_datasets()

history = neural_network(X, y, (32,32,32,32,32), n_iter= 100000, learning_rate=0.1)




# print(history)
# print('-----------------------\n')


# print(history[0][0], '\n----------\n', history[0][1], '\n----------\n', history[0][2],  '\n----------\n', history[0][3])





# print("dico iter ", iter)
# print('parametres :', history[0][0])
# print('----------')
# print('train_loss :', history[0][1])
# print('----------')
# print('train_acc :', history[0][2])
# print('----------')
# print('i :', history[0][3])

# print(parametres)
# print('----------')
# print(train_loss[iter])
# print('----------')

# print('----------')
# print(train_loss)
# print('----------')

# visualization(X, y, parametres, train_loss, train_accu, iter)








ax = np.array([], [])

def video(history):
	# print(history)
	# print(history[1])

	parametres = history[0]
	train_loss = history[1]
	train_accu = history[2]
	i = history[3]


	ax[0].clear() #frontiere de decision
	ax[1].clear() #sigmoide
	ax[2].clear() #fonction cout

	# s = 300
	resolution = 100



	# # Graph 1 -------------------------------------------------------------------------

	ax[0].set_title('Frontiere de Decision')
	ax[0].set_xlabel('x1')
	ax[0].set_ylabel('x2')
	ax[0].scatter(X[0, :], X[1, :], c=y, cmap='summer')

	x0_lim = ax[0].get_xlim()
	x1_lim = ax[0].get_ylim()
	x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
	x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)

	# # mshgrid

	X0, X1 = np.meshgrid(x0, x1)
	# print(X0.shape)
	# print(X0[:4, :4])

	#assemble (100, 100) -> 10000, 2

	XX = np.vstack((X0.ravel(), X1.ravel()))


	# print(XX.shape)

	Z = predict(XX, parametres)
	# print(Z.shape)

	Z = Z.reshape((resolution, resolution))

	ax[0].pcolormesh(X0, X1, Z, cmap='summer', alpha=0.3, zorder =-1)
	ax[0].contour(X0, X1, Z)

	# # Graph 2 -------------------------------------------------------------------------

	ax[1].plot(range(i), train_loss[:i], c='red', lw=2)

	ax[1].set_xlim(0, i)
	ax[1].set_ylim(0, train_loss[0] * 1.1)
	ax[1].set_title('Fonction Cout')
	ax[1].set_xlabel('iteration = ' + str(i))
	ax[1].set_ylabel('Loss')

	# # Graph 3 -------------------------------------------------------------------------

	ax[2].plot(range(i), train_accu[:i], c='red', lw=2)
	ax[2].set_xlim(0, i)
	ax[2].set_ylim(0, 1.1)
	ax[2].set_title('Fonction Cout')
	ax[2].set_xlabel('iteration = ' + str(i))
	ax[2].set_ylabel('Vraisemblence')
	return ax




fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40,10))
# fig, ax = plt.subplots()
ani = FuncAnimation(fig, video, frames=history, interval=200, repeat=False)

test = plt.show()

import matplotlib.animation as animation
# ani.save('neural_network.mp4')
