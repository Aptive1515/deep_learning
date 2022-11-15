# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    generator_datasets.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aptive <aptive@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/14 13:58:35 by aptive            #+#    #+#              #
#    Updated: 2022/11/14 14:01:25 by aptive           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from	sklearn.datasets import make_blobs, make_circles, make_moons



def	generator_datasets():
	# dataset
	# X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2)

	X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=100 )
	# X, y = make_moons(n_samples=100, noise=.05)

	X = X.T
	y = y.reshape((1, y.shape[0]))

	print('dimensions de X:', X.shape)
	print('dimensions de y:', y.shape)

	# plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')


	# plt.show()

	return X, y
