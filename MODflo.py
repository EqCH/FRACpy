# modflow model 6
# main copy github flopy - USGS

import sys
import os
import platform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
from flopy.utils.gridgen import Gridgen
from flopy.utils.reference import SpatialReference
import shapefile as sf #in case you dont have it, form anaconda prompt: pip install pyshp
import rasterio as ras
from rasterio.plot import show
from scipy.interpolate import griddata
import tools


# ____ ________________----------------------------
#### ## Parte 1 de fijar el modelo ##

# Model input files and output files will reside here.

model_name = 'Je_PR1'
model_ws = 'ext'


# Simulation  = TDIS & GWF model & IMS (Iterative model solution)


#//// create simulation
sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', exe_name='mf6',
                             sim_ws=model_ws)

#//// discretización temporal
#tdis_rc = [(1.0, 1, 1.0), (10.0, 120, 1.0),
#           (10.0, 120, 1.0), (10.0, 120, 1.0)]
tdis_rc = [(1.0, 1, 1.0)]


tdis = flopy.mf6.ModflowTdis(sim, pname='tdis', time_units='DAYS',
                             nper=1, perioddata=tdis_rc)

# calibration condition is condiered if assigned 1.0  stress preiod
# los datos temporales entran considerando  el:
# stress period, time step, and progresion =  t[i]/t[i+1]

#//// Create the Flopy groundwater flow (gwf) model object
gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name,
                           model_nam_file='{}.nam'.format(model_name))
gwf.name_file.save_flows = True


#//// Create the Flopy iterative model solver (ims) Package object

ims = flopy.mf6.ModflowIms(sim, pname='ims', print_option='SUMMARY',
                           complexity='SIMPLE', outer_hclose=1.e-5,
                           outer_maximum=100, under_relaxation='NONE',
                           inner_maximum=100, inner_hclose=1.e-6,
                           rcloserecord=0.1, linear_acceleration='BICGSTAB',
                           scaling_method='NONE', reordering_method='NONE',
                           relaxation_factor=0.99)
sim.register_ims_package(ims, [gwf.name])


#/// spatial discretization
# Open the shapefiles from the model limit and the refiment area around the wells Pilas que es para el caso que se
# tenga  los shapes para definir las ubicaciiones de interes

LimitShp = sf.Reader('gis/AreaE.shp')
LimitShp1 = sf.Reader('gis/AreaEdetallada.shp')

crs = {'init' :'epsg:3116'}
GloRefBox = LimitShp.bbox
LocRefBox = LimitShp1.bbox

#limitArray = np.array(LimitShp.shapeRecords()[0].shape.points) # para grafico


#Defining Global and Local Refinements, for purpose of simplicity cell x and y dimension will be the same

celGlo = 300
celRef = 400
# pilas tok cargar las funciones

delRArray = tools.arrayGeneratorCol(GloRefBox,LocRefBox,celGlo,celRef)
# se puede llamar el objeto para ver como esta

delCArray = tools.arrayGeneratorRow(GloRefBox, LocRefBox, celGlo, celRef)

#Calculating number or rows and cols since they are dependant from the discretization

nlay = 50
delz = -100
nrow = delCArray.shape[0]
ncol = delRArray.shape[0]
mtop = 0
botm = [delz for x in range(nlay)]

print('Number of rows: %d and number of cols: %d' % (nrow,ncol))

dis = flopy.mf6.ModflowGwfdis(gwf, pname='dis', nlay=nlay, nrow=nrow, ncol=ncol,
                              delr=delRArray, delc=delCArray, top=mtop,
                              botm=botm,
                              filename='{}.dis'.format(model_name))



gwf.dis.xorigin = GloRefBox[0]
gwf.dis.yorigin = GloRefBox[1]


gwf.modelgrid.set_coord_info(xoff=GloRefBox[0], yoff=GloRefBox[3]-delCArray.sum(), angrot=0,epsg=3116)




## read raster to xyz
fileDEM= 'gis/DEM_12m.tif'
raster = ras.open(fileDEM)
data = raster.read(1)
cell = raster.res[0]

limits=np.zeros((3,2))

limits[0,0] = raster.bounds.left + cell/2
limits[1,0] = raster.bounds.bottom +  cell/2
limits[0,1] = raster.bounds.right #- cell/2
limits[1,1] = raster.bounds.top #- cell/2

X=np.arange(limits[0,0],limits[0,1],cell)
Y=np.arange(limits[1,0],limits[1,1],cell)
X,Y = np.meshgrid(X,Y)
ou= X.size
points = np.array([X.reshape(ou),Y.reshape(ou)])
values = data.reshape(ou)


grid_x = gwf.modelgrid.xcellcenters
grid_y = gwf.modelgrid.ycellcenters


mtop = griddata(points.transpose(), values, (grid_x, grid_y), method='nearest')

# exportando txt con la malla para analisis de fracturas y transformación a un continuo
# falta adaptación para usar directamente .exe
"""
tec=np.zeros((mtop.size,3))
tec[:,0]=grid_x.reshape(mtop.size)
tec[:,1]=grid_y.reshape(mtop.size)
tec[:,2]=mtop.reshape(mtop.size)

file='txt/p1.txt'
tools.writeNumpy(tec,file) ### yo creo que se puede llamar directamente a .exe
"""
#fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(1, 1, 1)
#ax.set_aspect('equal')
#modelmap = flopy.plot.PlotMapView(model=gwf,ax=ax)
#quadmesh = modelmap.plot_array(mtop)
#linecollection = modelmap.plot_grid(linewidth=0.5, color='royalblue')
#plt.show()


gwf.dis.top  = mtop
zbot = np.zeros((nlay,nrow,ncol))
for i in range(nlay-1):
    zbot[i,:,:] = mtop+i*botm[i]*2

zbot[(nlay-1),:,:]  = (zbot[i,:,:].min())-50
#Asign layer bottom elevations

gwf.dis.botm = zbot


#for i in range(5):
#    fig = plt.figure(figsize=(20, 3))
#    ax = fig.add_subplot(1, 1, 1)
#    modelxsect = flopy.plot.PlotCrossSection(model=gwf, line={'Row': i*3})
#    linecollection = modelxsect.plot_grid()
#    plt.show()

# iniciando llenado y generado de modelo de conductividades

file = "ReadFortran/imtemp.dat"
xyz, cvec = tools.readfile(file,5)

zbot = 0.0
for i in range(nlay-1):

    layer =  (xyz[:,3] == i+1)
    zbot[i,:,:] = np.array(xyz[layer,4].reshape(nrow,ncol),dtype=np.float64)

layer =  (xyz[:,3] == nlay)
zbot[(nlay-1),:,:]  = xyz[layer,4].reshape(nrow,ncol)
