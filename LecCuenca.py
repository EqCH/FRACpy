import rasterio as ras
from rasterio.plot import show
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
#matplotlib.use("TkAgg")
from  matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


f = open("data/model.read","r").read().split()
files = 'data/'+f[0].strip()
f = open(files,"r").read().split("\n")
print("initial model")
f.remove('')



limits=np.zeros((3,2))



# Raster
fileDEM = 'goe/'+f[1].strip()+'/'+'DEM_12m.tif'
raster = ras.open(fileDEM)
data = raster.read(1)
cell = raster.res[0]

limits[0,0] = raster.bounds.left + cell/2
limits[1,0] = raster.bounds.bottom +  cell/2
limits[0,1] = raster.bounds.right #- cell/2
limits[1,1] = raster.bounds.top #- cell/2

X=np.arange(limits[0,0],limits[0,1],cell)
Y=np.arange(limits[1,0],limits[1,1],cell)
X,Y = np.meshgrid(X,Y)

ou = 9+ int(f[6])
for i in range(3):
	print('valores de i ', i)
	limits[i,0] = float(f[ou+2*i].strip().split(" ")[0])
	limits[i,1] = float(f[ou+2*i].strip().split(" ")[1])


# lectura de discos generados por fracture o main o Tflow
# frac

file = "out/"+f[-1]+".disk"
Frac = pd.read_csv(file,header=0)



tec=Frac.index[Frac['x'].isnull()].tolist()
id1=[-1]
id1.extend(tec)



## dicos generados ##

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

#  Plot the surface.

#surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)

# Add a color bar which maps values to colors.

# Discos

for i in range(len(id1)):
	id1[i]=id1[i]+1
	x = list( Frac[id1[i]:id1[i]+140]['x'])
	y = list( Frac[id1[i]:id1[i]+140]['y'])
	z = list( Frac[id1[i]:id1[i]+140]['z'])
	for j in range(len(x)):
		x[j] = float(x[j])
		y[j] = float(y[j])
		z[j] = float(z[j])
	verts = [list(zip(x,y,z))]
	col1 =list(np.random.rand(3))
	col1.append(0.05)
	ax.add_collection3d(Poly3DCollection(verts,facecolors=col1,linestyle='solid'))
	ax.plot(x,y,z,color='m',alpha=0.2)

# Customize the z axis.

ax.set_xlim(limits[0,0], limits[0,1])
ax.set_ylim(limits[1,0], limits[1,1])
ax.set_zlim(250., 3635.)

file1 = "out/"+f[-1]+".dat"
Inter =  pd.read_csv(file1)
Inter = np.array(Inter)


for i in range(Inter.shape[0]-1):
	x = Inter[i,[0,3]]
	y = Inter[i,[1,4]]
	z = Inter[i,[2,5]]
	plt.plot(x,y,z,color='r',alpha=0.15)


plt.show()
