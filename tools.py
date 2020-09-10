import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import os

#  exec(open("filename.py").read())

def arrayGeneratorCol(gloRef, locRef, gloSize, locSize):

    cellArray = np.array([])

    while cellArray.sum() + gloRef[0] < locRef[0] -  gloSize:
        cellArray = np.append(cellArray,[gloSize])
    while cellArray.sum() + gloRef[0] > locRef[0] -  gloSize and cellArray.sum() + gloRef[0] < locRef[2] +  gloSize:
        cellArray = np.append(cellArray,[locSize])
    while cellArray.sum() + gloRef[0] > locRef[2] +  gloSize and cellArray.sum() + gloRef[0] < gloRef[2]:
        cellArray = np.append(cellArray,[gloSize])

    return cellArray
def arrayGeneratorRow(gloRef, locRef, gloSize, locSize):

    cellArray = np.array([])
    accumCoordinate =  gloRef[3] - cellArray.sum()

    while gloRef[3] - cellArray.sum() > locRef[3] +  gloSize:
        cellArray = np.append(cellArray,[gloSize])
    while gloRef[3] - cellArray.sum() < locRef[3] +  gloSize and gloRef[3] - cellArray.sum() > locRef[1] -  gloSize:
        cellArray = np.append(cellArray,[locSize])
    while gloRef[3] - cellArray.sum() < locRef[1] -  gloSize and gloRef[3] - cellArray.sum() > gloRef[1]:
        cellArray = np.append(cellArray,[gloSize])

    return cellArray

## generador de imagenes para ejevutar desde consola ##

def plotmodel(gwf,k):
    gwf.check()
    fig = plt.figure(figsize=(20, 3))
    ax = fig.add_subplot(1, 1, 1)
    modelxsect = flopy.plot.PlotCrossSection(model=gwf, line={'Row': 20})
    linecollection = modelxsect.plot_grid()
    modelxsect.plot_array(k)

def ploCHD(Nlay,N,ra):
    # We camake a quick plot to show where our constant
    # heads are located by creating an integer array
    # that starts with ones everywhere, but is assigned
    # a -1 where chds are located
    ibd = np.ones((Nlay, N, N), dtype=np.int)
    for k, i, j in ra['cellid']:
        ibd[k, i, j] = -1

    ilay = 0
    plt.imshow(ibd[ilay, :, :], interpolation='none')
    plt.title('Layer {}: Constant Head Cells'.format(ilay + 1))


def plotheads(workspace,headfile,L,N):
    # Read the binary head file and plot the results
    # We can use the existing Flopy HeadFile class because
    # the format of the headfile for MODFLOW 6 is the same
    # as for previous MODFLOW verions
    fname = os.path.join(workspace, headfile)
    hds = flopy.utils.binaryfile.HeadFile(fname)
    h = hds.get_data(kstpkper=(0, 0))
    x = y = np.linspace(0, L, N)
    y = y[::-1]
    c = plt.contour(x, y, h[0], np.arange(90,100.1,0.2))
    plt.clabel(c, fmt='%2.1f')
    plt.axis('scaled')


def writeNumpy(tec,file):
    a_file=open(file,'w')
    for raw in tec:
        a_file.write('{: f} {: f} {: f}\r\n'.format(raw[0],raw[1],raw[2]))
    a_file.close()


def readfile(file,col):
	import numpy as np
	a_file=open(file,'r')
	aline=a_file.readlines()
	a_file.close()
	xyz = np.zeros((len(aline),col))
	for i in range(len(aline)):
		for j in range(5):
			xyz[i,j]=np.array(aline[i].split()[j],dtype=np.float64)
        vec[i] = integer(xyz[i,4])
	return xyz, vec
