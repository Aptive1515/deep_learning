# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    video.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/14 14:13:06 by aptive            #+#    #+#              #
#    Updated: 2022/11/14 14:14:33 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from neural_network import *
from matplotlib.animation import FuncAnimation

def video(X, y, history):
	ax = np.array([], [])

	def anime(params):
		W = params[0]
		b = params[1]
		loss = params[2]
		i = params[3]

		ax[0].clear() #frontiere de decision
		ax[1].clear() #sigmoide
		ax[2].clear() #fonction cout

		s = 300
		#frontiere de decision
		ax[0].scatter(X[:,0], X[:,1], c=y,s=s, cmap ="summer", edgecolor="k", linewidths=1)

		xlim = ax[0].get_xlim()
		ylim = ax[0].get_ylim()

		x1 = np.linspace(-3, 6, 100)
		x2 = (-W[0] * x1 - b) / W[1]
		ax[0].plot(x1, x2, c = 'orange', lw= 4)

		ax[0].set_xlim(X[:,0].min(), X[:,0].max())
		ax[0].set_ylim(X[:,1].min(), X[:,1].max())

		ax[0].set_title('Frontiere de Decision')
		ax[0].set_xlabel('x1')
		ax[0].set_ylabel('x2')

		# #sigmoide

		z = X.dot(W) + b
		z_new = np.linspace(z.min(), z.max(), 100)
		A = 1 / (1 + np.exp(-z_new))

		ax[1].plot(z_new, A, c='orange', lw=4)

		ax[1].scatter(z[y==0], np.zeros(z[y==0].shape), c ='#008066', edgecolors='k', linewidths=3, s=s)
		ax[1].scatter(z[y==1], np.ones(z[y==1].shape), c ='#ffff66', edgecolors='k', linewidths=3, s=s)

		ax[1].set_xlim(z.min(), z.max())
		ax[1].set_title('Sigmoide')
		ax[1].set_xlabel('Z')
		ax[1].set_ylabel('A(Z)')

		for j in range(len(A[y.flatten()==0])):
			ax[1].vlines(z[y==0][j], ymin=0, ymax=1 / (1 + np.exp(-z[y==0][j])), color='red', alpha=0.5, zorder =-1)

		for j in range(len(A[y.flatten()==1])):
			ax[1].vlines(z[y==1][j], ymin=1, ymax=1 / (1 + np.exp(-z[y==1][j])), color='red', alpha=0.5, zorder =-1)

		ax[2].plot(range(i), loss[:i], c='red', lw=4)
		ax[2].set_xlim(loss[-1] * 0.8, len(loss))
		ax[2].set_ylim(0, loss[0] * 1.1)
		ax[2].set_title('Fonction Cout')
		ax[2].set_xlabel('iteration = ' + str(i) + '\n' + 'Vraisemblence = ' + str((loss[i])))
		ax[2].set_ylabel('Loss')
		return ax

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40,10))
	# fig, ax = plt.subplots()
	ani = FuncAnimation(fig, anime, frames=history, interval=200, repeat=False)
