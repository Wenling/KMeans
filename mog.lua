--[[
Mixture of Gaussians Implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The mixture of gaussians algorithm should be presented here. You can implement
it in anyway you want. For your convenience, a multivariate gaussian object is
provided at gaussian.lua.

Here is how I implemented it:

mog(n,k) is a constructor to return an object m which will perform MoG
algorithm on data of dimension n with k gaussians. The m object stores the i-th
gaussian at m[i], which is a gaussian object. The m object has the following
methods:

m:g(x): The decision function which returns a vector of k elements indicating
each gaussian's likelihood

m:f(x): The output function to output a prototype that could replace vector x.

m:learn(x,p,eps): learn the gaussians using x, which is an m*n matrix
representing m data samples. p is regularization to keep each gaussian's
covariance matrices non-singular. eps is a stop criterion.
]]

dofile("gaussian.lua")
dofile("kmeans.lua")

-- Create a MoG learner
-- n: dimension of data
-- k: number of gaussians
function mog(n,k)
	local m = {}
	m.features = n
	m.gaussian_size = k
	m.datasize = 0
	m.R = torch.Tensor(1, 1):zero() -- Responsibilities
	m.W = torch.Tensor(1):zero() -- Weights

	-- init
	for i = 1, m.gaussian_size do
		m[i] = gaussian(m.features)
	end

	-- P(X | W1...Wk,M1...Mk,A1...AK) = \sum_j Wj*G(Xi,Mj,Aj)
	function m:g(x)
		local P = torch.zeros(m.gaussian_size)
		for j = 1, m.gaussian_size do
			P[j] = m[j]:eval(x) * m.W[j]
		end
		-- normalize
		local sum = torch.sum(P)
		P:div(sum)
		return P
	end

	function m:f(x)
		local y, index = torch.max(m:g(x), 1)
		return m[index[1]].m
	end

	function m:learn(X, p, eps)
		-- init with KMeans
		m.datasize = X:size(1)
		local km = kmeans(m.features, m.gaussian_size)
		km:learn(X)
		m.R:resize(m.datasize, m.gaussian_size)
		m.R:copy(km.R)
		-- init gaussian M, A
		m:mstep(X, p)
--		print(m.W)
		local diff0 = m:cvg(X)
		print("L="..diff0)
		local diff1 = 0
		local epoch = 10
		for i = 1, epoch do
			m:estep(X)
         	m:mstep(X,p)
         	diff1 = m:cvg(X)
         	print("diff="..(diff0 - diff1) / diff0)
         	-- converged
         	if torch.abs((diff0 - diff1) / diff0) < eps then
         		break
         	-- diverged
         	elseif (diff0 - diff1) / diff0 <= 0 then
         		print("diverged..")
         		break 
         	end
         	diff0 = diff1
		end
		print("L="..diff0)
	end

	--E step
	function m:estep(X)
		for i = 1, m.datasize do
			m.R:select(1,i):copy(m:g(X[i]))
		end
	end

	--M step
	function m:mstep(X, p)
		local sum = torch.sum(m.R, 1)[1]
		for i = 1, m.gaussian_size do
			m[i]:learn(X, m.R:select(2,i), p)
		end
		-- \sum j is 1
		m.W = torch.div(sum, m.datasize)
	end

	-- compute the convergence
	-- neg log likelihood around the maximum likelihood estimate
	function m:cvg(X)
		local err = 0
		for i = 1, m.datasize do
			err = err - torch.sum(torch.log(m:g(X[i])))
--			print("log="..err)
		end
		return err
	end
	
	function m:compress(X)
		local X_new = torch.zeros(X:size())
		local datasize = X:size()[1]
		for i = 1, datasize do
			X_new[i] = m:f(X[i]):clone()
		end
		return X_new
	end
	
	-- loss
	function m:loss(X, M)
		local loss = 0
		local datasize = X:size()[1]
		for i = 1, datasize do
			-- avg (||Xi - Tk(i)||^2)
			loss = loss * ((i - 1) / i) + torch.norm(X[i] - M[i])^2 / i
		end
		return loss
	end	
	return m
end
