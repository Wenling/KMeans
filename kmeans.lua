--[[
K-Means clustering algorithm implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The k-means algorithm should be presented here. You can implement it in any way
you want. For your convenience, a clustering object is provided at mcluster.lua

Here is how I implemented it:

kmeans(n,k) is a constructor to return an object km which will perform k-means
algorithm on data of dimension n with k clusters. The km object stores the i-th
cluster at km[i], which is an mcluster object. The km object has the following
methods:

km:g(x): the decision function to decide which cluster the vector x belongs to.
Return a scalar representing cluster index.

km:f(x): the output function to output a prototype that could replace vector x.

km:learn(x): learn the clusters using x, which is a m*n matrix representing m
data samples.
]]

dofile("mcluster.lua")

-- Create a k-means learner
-- n: dimension of data
-- k: number of clusters
function kmeans(n,k)
	-- The km object stores the i-th cluster at km[i]
	local km = {}
	km.features = n
	km.cluster_size = k
	km.datasize = 0 -- # of tiles
	km.R = torch.zeros(1, 1) -- Responsibilities

	-- init km
	for i = 1, km.cluster_size do
		km[i] = mcluster(km.features)
	end

	function km:min_dist(x)
		local distance = torch.ones(km.cluster_size)
		for i = 1, km.cluster_size do
			distance[i] = km[i]:eval(x)
		end
		local y, index = torch.min(distance, 1)
		return index[1]
	end

	-- Decision function
	-- Return a scalar representing cluster index.
	function km:g(x)
		return km:min_dist(x)
	end

	-- Output function
	function km:f(x)
		return km[km:g(x)].m
	end

	-- Learn the clusters using x
	function km:learn(X)
		local epoch = 100
		local datasize = X:size()[1] -- set datasize
		km.R:resize(datasize, km.cluster_size):zero() -- Responsibility matrix, 1 if Xi belongs to Rij, and 0 else

		-- init clusters
		local rand = torch.randperm(datasize)
		for i = 1, km.cluster_size do
			km[i]:set_m(X[rand[i]])
		end
		-- repeat until converge
		for k = 1, epoch do
			local converged = true
			-- compute center of each sample
			for i = 1, datasize do
				local new_center = km:g(X[i])
				if km.R[i][new_center] ~= 1 then
					converged = false
					km.R[i]:fill(0) -- this is slow
					km.R[i][new_center] = 1
				end
			end
			if converged == true then
				break
			end
			-- update cluster
			for j = 1, km.cluster_size do
				-- one cluster converge to one point (dist to itself is 0)
				if torch.sum(km.R:select(2,j)) <= 0 then
					print("one cluster converge to one point")
					km[j]:set_m(X[torch.randperm(datasize)[1]])
				else
					km[j]:learn(X, km.R:select(2, j))
				end
			end
			print("iter=".. k)
		end
	end

	-- compress image
	function km:compress(X)
		local X_new = torch.zeros(X:size())
		local datasize = X:size()[1]
		for i = 1, datasize do
			X_new[i] = km:f(X[i]):clone()
		end
		return X_new
	end

	-- loss
	function km:loss(X, M)
		local loss = 0
		local datasize = X:size()[1]
		for i = 1, datasize do
			-- avg (||Xi - Tk(i)||^2)
			loss = loss * ((i - 1) / i) + torch.norm(X[i] - M[i])^2 / i
		end
		return loss
	end	
	
	-- Number of bits
	function km: compress_rate(X)
		local datasize = X:size()[1]
		local H = torch.zeros(km.cluster_size)
		local num_in_cluster = torch.zeros(km.cluster_size) -- number of samples in each cluster
		for k = 1, km.cluster_size do
			num_in_cluster[k] = torch.sum(km.R:select(2, k)) 
		end
		for k = 1, km.cluster_size do
			H[k] = num_in_cluster[k] / datasize
		end
		local Histogram_Entropy = -torch.sum(H:cmul(torch.log(H):div(torch.log(2))))
		local Number_of_bits = datasize * Histogram_Entropy
		return Number_of_bits
	end

	return km
end
