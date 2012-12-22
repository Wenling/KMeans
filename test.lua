require("image")
dofile("tile.lua")
dofile("kmeans.lua")
dofile("mog.lua")

-- An example of using tile
function Q21(t, K)
	

	local kt = t:clone()
	local km = kmeans(64, K)
	km:learn(t)
	local X_new = km:compress(kt)
	local loss = km:loss(kt, X_new)
	print("K="..K.."  Compress_loss="..loss)
	return km, X_new
end

function Q22()
	local img = 'boat.png'
	-- Read file
	im = tile.imread(img)
	-- Convert to 7500*64 tiles representing 8x8 patches
	t = tile.imtile(im,{8,8})
	
	local K = tonumber(arg[1])
	local km, X_new = Q21(t, K)

	-- Convert back to 800*600 image with 8x8 patches
	im2 = tile.tileim(X_new,{8,8},{600,800})
	-- Show the image
	image.display(im2)
	-- The following call can save the image
	tile.imwrite(im2,'boat2.png')
end

function Q23()
	local img = 'boat.png'
	-- Read file
	im = tile.imread(img)
	-- Convert to 7500*64 tiles representing 8x8 patches
	t = tile.imtile(im,{8,8})
	local K = tonumber(arg[1])
	local p = tonumber(arg[2])
	local eps = tonumber(arg[3])
	local t_mog = t:clone()
	local mixg = mog(64, 8)
	mixg:learn(t, p, eps)
	
	local XX_new = mixg:compress(t_mog)
	local loss = mixg:loss(t_mog, XX_new)
	print("K="..K.."  Compress_loss="..loss)
	
	-- Convert back to 800*600 image with 8x8 patches
	im2 = tile.tileim(XX_new,{8,8},{600,800})
	-- Show the image
	image.display(im2)
	-- The following call can save the image
	tile.imwrite(im2,'boat3.png')
end

--Q22()
Q23()