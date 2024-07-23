#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lasio
import seaborn as sns


# In[ ]:


import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


# In[ ]:


import requests
import xarray as xr

# Machine Learning Packages
import sklearn
from sklearn import datasets
from sklearn.preprocessing import scale

import minisom
from minisom import MiniSom


# In[ ]:


torch.__version__ #it should be '1.9.1+cpu'

#see requirements in the README


# ## Data

# In[ ]:


las= lasio.read('Sonic_P_and_S_Output_7240ft_350ft.las')


# In[ ]:


for count, curve in enumerate(las.curves):
    print(f"Curve: {curve.mnemonic}, \t Units: {curve.unit}, \t Description: {curve.descr}")
print(f"There are a total of: {count+1} curves present within this file")


# In[ ]:


well = las.df()
well.head()


# In[ ]:


well.plot(y='RHOZ')


# In[ ]:


well_clean = well.dropna(subset=['VPVS','RHOZ', 'DTCO','DTSM','SPHI','DPHZ','GR_EDTC'])
depth = well_clean.index.values


# ## Smoothening velocidades

# In[ ]:


#velocidades
win=101

print(well_clean['DTCO'].values.shape)
zz_p_smooth = signal.savgol_filter(well_clean['DTCO'].values, win, 3)
zz_s_smooth = signal.savgol_filter(well_clean['DTSM'].values, win, 3)


# In[ ]:


fig = make_subplots(rows=2, cols=1, shared_yaxes=True, subplot_titles=( "Slowness s wave", "Slowness p wave"))

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth,
    y=well_clean['DTSM'].values,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
),row=1, col=1)
fig.add_trace(go.Scatter(
    x=depth,
    y=zz_s_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=1, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=depth,
    y=well_clean['DTCO'].values,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)
fig.add_trace(go.Scatter(
    x=depth,
    y=zz_p_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)



# Ajustar el diseño
#fig.update_xaxes(range=[395, 5600], row=1, col=1)
#fig.update_xaxes(r row=2, col=1)
fig.update_xaxes(title_text="Depth", row=2, col=1)
# Mostrar la figura
fig.show()


# In[ ]:


#de sl a vel
vs = (1/ well_clean['DTSM'].values)*0.3048*(10**6)
vp = (1/ well_clean['DTCO'].values)*0.3048*(10**6)

vs_smooth = (1/zz_s_smooth)*0.3048*(10**6)
vp_smooth = (1/zz_p_smooth)*0.3048*(10**6)

depth_m = depth*0.3048


# In[ ]:


fig = make_subplots(rows=2, cols=1, shared_yaxes=True, subplot_titles=( "Vs", "Vp"))

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vs,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
),row=1, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vs_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=1, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vp,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vp_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)



# Ajustar el diseño
fig.update_yaxes(title_text="velocity (m/s)", row=1, col=1)
fig.update_yaxes(title_text="velocity (m/s)", row=2, col=1)
fig.update_xaxes(title_text="Depth (m)", row=2, col=1)
# Mostrar la figura
fig.show()


# ## Smoothening porosity and density

# In[ ]:


# SPHI, DPHZ, RHOZ
print(well_clean['SPHI'].values.shape)
sonicPor_smooth = signal.savgol_filter(well_clean['SPHI'].values, win, 3)
son = well_clean['SPHI'].values

print(well_clean['DPHZ'].values.shape)
DensPor_smooth = signal.savgol_filter(well_clean['DPHZ'].values, 55, 3)
densp = well_clean['DPHZ'].values

print(well_clean['RHOZ'].values.shape)
Dens_smooth = signal.savgol_filter(well_clean['RHOZ'].values, win, 3)
dens = well_clean['RHOZ'].values


# In[ ]:


#units to SI
# porosity is dimensionless so it's ok, Dens kg/m^3

Dens_kgm3_sm = Dens_smooth*1000
Dens_kgm3 = well_clean['RHOZ'].values*1000


# In[ ]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=( "Sonic Porosity", "Density porosity", "Density"))

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=son,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
),row=1, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=sonicPor_smooth,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=1, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=densp,  
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=DensPor_smooth,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)

# Densidad
fig.add_trace(go.Scatter(
    x=depth_m,
    y=Dens_kgm3,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=Dens_kgm3_sm,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)

# Ajustar el diseño
fig.update_yaxes(title_text="Sonic Porosity", row=1, col=1)
fig.update_yaxes(title_text="Porosity",range=[-0.200,1.00], row=2, col=1)
fig.update_yaxes(title_text="$Density (kg/m^3)$", row=3, col=1)
fig.update_xaxes(title_text="Depth (m)", row=3, col=1)

fig.update_xaxes(range=[119,1200], row=2, col=1)
# Mostrar la figura
fig.show()


# In[ ]:


sns.boxplot(densp)


# In[ ]:


df=well_clean[['DPHZ']].copy()


# In[ ]:


#por si acaso xd
Q1 = df['DPHZ'].quantile(0.25)
Q3 = df['DPHZ'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites para identificar valores atípicos
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrar los valores atípicos
filtered_df = df[(df['DPHZ'] >= lower_bound) & (df['DPHZ'] <= upper_bound)]

# Graficar el diagrama de caja sin los valores atípicos
sns.boxplot(data=filtered_df['DPHZ'])
plt.show()


# In[ ]:


filtered_df2 = df[(df['DPHZ'] <= 1.2)]

# Graficar el diagrama de caja sin los valores atípicos
sns.boxplot(data=filtered_df2['DPHZ'])
plt.show()


# In[ ]:


sm_pd=signal.savgol_filter(filtered_df2['DPHZ'].values, 201, 5)


# In[ ]:


# Crear la subtrama
fig = make_subplots(rows=1, cols=1, shared_yaxes=True)

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=filtered_df2['DPHZ'].values,
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
), row=1, col=1)

# Añadir la segunda subtrama (comentada)

fig.add_trace(go.Scatter(
    x=depth_m,
    y=sm_pd,  # Asegúrate de que 'sonicPor_smooth' esté definido
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
), row=1, col=1)


# Ajustar el diseño
fig.update_yaxes(title_text="Porosity (fraction)", row=1, col=1)
fig.update_xaxes(title_text="Depth (m)", row=1, col=1)

fig.update_layout(
    title_text='Density porosity',
    title_x=0.5,  # Centrar el título
    title_y=0.9,  # Ajustar la posición vertical del título
)

# Mostrar la figura
fig.show()


# In[ ]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=( "Vs", "Vp","Density"))

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vs,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=True
),row=1, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vs_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=True
),row=1, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vp,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vp_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)

# Densidad
fig.add_trace(go.Scatter(
    x=depth_m,
    y=Dens_kgm3,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=Dens_kgm3_sm,  # Ventana de tamaño 101 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)


# Ajustar el diseño
fig.update_yaxes(title_text="velocity (m/s)", row=1, col=1)
fig.update_yaxes(title_text="velocity (m/s)", row=2, col=1)
fig.update_yaxes(title_text="$Density (kg/m^3)$", row=3, col=1)
fig.update_xaxes(title_text="Depth (m)", row=3, col=1)

fig.update_layout(title_text="Comparison of Geophysical Properties Before and After Smoothing Across Depth")
# Mostrar la figura
fig.show()


# ## Smoothening gamma ray and Vp/Vs

# In[ ]:


#Smoothening gamma ray
win2=85

GR_sm=signal.savgol_filter(well_clean['GR_EDTC'].values, 51, 3)

vpvs_sm=signal.savgol_filter(well_clean['VPVS'].values, win2, 3)


# In[ ]:


# Crear la subtrama
fig = make_subplots(rows=2, cols=1, shared_yaxes=True, subplot_titles=( "Gamma ray", "Vp/Vs"))
# Añadir la primera subtrama (GR)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=well_clean['GR_EDTC'].values,
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
), row=1, col=1)

# Añadir la segunda subtrama (comentada)

fig.add_trace(go.Scatter(
    x=depth_m,
    y=GR_sm,  # Asegúrate de que 'sonicPor_smooth' esté definido
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
), row=1, col=1)


# Añadir la primera subtrama (Vp/Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=well_clean['VPVS'].values,
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
), row=2, col=1)

# Añadir la segunda subtrama (comentada)

fig.add_trace(go.Scatter(
    x=depth_m,
    y=vpvs_sm,  # Asegúrate de que 'sonicPor_smooth' esté definido
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
), row=2, col=1)

# Ajustar el diseño
fig.update_yaxes(title_text="Gamma ray (GAPI)", row=1, col=1)
fig.update_xaxes(title_text="Depth (m)", row=1, col=1)

fig.update_yaxes(title_text="Vp/Vs ", row=2, col=1)
fig.update_xaxes(title_text="Depth (m)", row=2, col=1)



# Mostrar la figura
fig.show()


# ## Filter density

# In[ ]:


sns.boxplot(dens)


# In[ ]:


df_d=well_clean[['RHOZ']].copy()


# In[ ]:


#por si acaso xd
Q1 = df_d['RHOZ'].quantile(0.25)
Q3 = df_d['RHOZ'].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites para identificar valores atípicos
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrar los valores atípicos
filtered_df_d = df_d[(df_d['RHOZ'] >= lower_bound) & (df_d['RHOZ'] <= upper_bound)]

# Graficar el diagrama de caja sin los valores atípicos
sns.boxplot(data=filtered_df_d['RHOZ'])
plt.show()


# In[ ]:


dens_pd=signal.savgol_filter(filtered_df_d['RHOZ'].values, 201, 5)*1000


# In[ ]:


# Crear la subtrama
fig = make_subplots(rows=1, cols=1, shared_yaxes=True)

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=filtered_df_d['RHOZ'].values*1000,
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
), row=1, col=1)

# Añadir la segunda subtrama (comentada)

fig.add_trace(go.Scatter(
    x=depth_m,
    y=dens_pd,  # Asegúrate de que 'sonicPor_smooth' esté definido
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
), row=1, col=1)


# Ajustar el diseño
fig.update_yaxes(title_text="density (kg/m^3)", row=1, col=1)
fig.update_xaxes(title_text="Depth (m)", row=1, col=1)

fig.update_layout(
    title_text='Density',
    title_x=0.5,  # Centrar el título
    title_y=0.9,  # Ajustar la posición vertical del título
)

# Mostrar la figura
fig.show()


# ## 2D model with the new filters

# In[ ]:


Nx = 10000
Nz_vp = int(vp_smooth.size)
vp2d = np.zeros([Nx,Nz_vp])
for i in range(0,Nx):
    vp2d[i] = vp_smooth


# In[ ]:


Nx = 10000
Nz_vs = int(vs_smooth.size)
vs2d = np.zeros([Nx,Nz_vs])
for i in range(0,Nx):
    vs2d[i] = vs_smooth


# In[ ]:


Nx = 10000
Nz_dens = int(Dens_kgm3_sm.size)
dens2d = np.zeros([Nx,Nz_dens])
for i in range(0,Nx):
    dens2d[i] = Dens_kgm3_sm


# In[ ]:


fig, axs = plt.subplots(3, 1, figsize=(10, 15))  

# Primer gráfico 250,5
im3 = axs[0].imshow(vs2d.T, cmap="copper")
axs[0].invert_yaxis()
cbar = fig.colorbar(im3, ax=axs[0])  
cbar.set_label("S velocity values (m/s)")

# Segundo gráfico 250,3
im2 = axs[1].imshow(vp2d.T, cmap="copper")
axs[1].invert_yaxis()
cbar = fig.colorbar(im2, ax=axs[1])
cbar.set_label(" P velocity values (m/s)")

# tercer gráfico 250,3
im1 = axs[2].imshow(dens2d.T, cmap="copper", vmax=3000)
axs[2].invert_yaxis()
cbar = fig.colorbar(im1, ax=axs[2])
cbar.set_label("$Density (kg/m^{3})$")

plt.show()


# In[ ]:


num_pixels = vs2d.shape[0]
depth_tot = max(depth_m)

print(str(depth_tot)+' m is the maximum depth')


# In[ ]:


depth_per_pixel = depth_tot / num_pixels
depth_per_pixel


# In[ ]:


depths = np.arange(0, depth_tot, depth_per_pixel)
depths


# In[ ]:


dens2d.T.shape


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(25, 8))

# Primer gráfico: S velocity values
im3 = axs[0].imshow(vs2d, cmap="copper", extent=[0, vs2d.shape[1], 0, depth_tot], aspect='auto')
axs[0].invert_yaxis()  # Invertir el eje y para que la profundidad aumente hacia abajo
cbar = fig.colorbar(im3, ax=axs[0])
cbar.set_label("s wave velocity values (m/s)")
axs[0].set_xticks([])  # Ocultar marcas del eje x
axs[0].set_xlabel('')  # Ocultar etiqueta del eje x

# Segundo gráfico: P velocity values
im2 = axs[1].imshow(vp2d, cmap="copper", extent=[0, vp2d.shape[1], 0, depth_tot], aspect='auto')
axs[1].invert_yaxis()  # Invertir el eje y para que la profundidad aumente hacia abajo
cbar = fig.colorbar(im2, ax=axs[1])
cbar.set_label("p wave velocity values (m/s)")
axs[1].set_xticks([])  # Ocultar marcas del eje x
axs[1].set_xlabel('')  # Ocultar etiqueta del eje x

# tercer gráfico 250,3
im1 = axs[2].imshow(dens2d, cmap="copper", extent=[0, vp2d.shape[1], 0, depth_tot], aspect='auto', vmax=3000)
axs[2].invert_yaxis()
cbar = fig.colorbar(im1, ax=axs[2])
cbar.set_label("$Density (kg/m^{3})$")
axs[2].set_xticks([])  # Ocultar marcas del eje x
axs[2].set_xlabel('')  # Ocultar etiqueta del eje x

# Etiquetas y título
axs[0].set_ylabel('Depth (m)')
axs[1].set_ylabel('Depth (m)')
axs[2].set_ylabel('Depth (m)')

fig.suptitle("preliminar subsurface model", fontsize=16)

plt.show()


# In[ ]:


#data
m_vs=vs2d.T
m_vp=vp2d.T
m_dens=dens2d.T


# In[ ]:


m_vs.astype(np.float32).tofile('Vs2d_shape_9410_10000.dat')
m_vp.astype(np.float32).tofile('Vp2d_shape_9410_10000.dat')
m_dens.astype(np.float32).tofile('Density2d_shape_9410_10000.dat')


# In[ ]:


# Cargar los archivos binarios
vs2d = np.fromfile('Vs2d_shape_9410_10000.dat', dtype=np.float32).reshape((9410, 10000))
vp2d = np.fromfile('Vp2d_shape_9410_10000.dat', dtype=np.float32).reshape((9410, 10000))
dens2d = np.fromfile('Density2d_shape_9410_10000.dat', dtype=np.float32).reshape((9410, 10000))

# Verificar algunas estadísticas básicas para asegurarse de que los datos se han guardado correctamente
print('vs2d mean:', np.mean(vs2d))
print('vp2d mean:', np.mean(vp2d))
print('dens2d mean:', np.mean(dens2d))


# ## 2D of every log used for the clustering

# In[ ]:


#por,son por, vpvs, gamma ray 
#the others are alredy processed :)

Nx = 10000

#Sonic Porosity (son and sonicPor_smooth)
Nz_SP = int(sonicPor_smooth.size)
SP2d = np.zeros([Nx,Nz_SP])
for i in range(0,Nx):
    SP2d[i] = sonicPor_smooth

#Density Porosity (DensPor_smooth)
Nz_DP = int(sm_pd.size)
DP2d = np.zeros([Nx,Nz_DP])
for i in range(0,Nx):
    DP2d[i] = sm_pd
    
#Gamma ray (GR_sm)
Nz_Gr = int(GR_sm.size)
Gr2d = np.zeros([Nx,Nz_Gr])
for i in range(0,Nx):
    Gr2d[i] = GR_sm

#Vp/Vs (vpvs_sm)
Nz_VpVs = int(vpvs_sm.size)
VpVs2d = np.zeros([Nx,Nz_VpVs])
for i in range(0,Nx):
    VpVs2d[i] = vpvs_sm


# In[ ]:


fig_t, axs_t = plt.subplots(3, 2, figsize=(20, 20))

# Primer gráfico: S velocity values
im1 = axs_t[0,0].imshow(vs2d, cmap="copper", extent=[0, vs2d.shape[1], 0, depth_tot], aspect='auto')
axs_t[0,0].invert_yaxis()  # Invertir el eje y para que la profundidad aumente hacia abajo
cbar = fig.colorbar(im1, ax=axs_t[0,0])
cbar.set_label("s wave velocity values (m/s)")
axs_t[0,0].set_xticks([])  # Ocultar marcas del eje x
axs_t[0,0].set_xlabel('')  # Ocultar etiqueta del eje x

# Segundo gráfico: P velocity values
im2 = axs_t[1,0].imshow(vp2d, cmap="copper", extent=[0, vp2d.shape[1], 0, depth_tot], aspect='auto')
axs_t[1,0].invert_yaxis()  # Invertir el eje y para que la profundidad aumente hacia abajo
cbar = fig.colorbar(im2, ax=axs_t[1,0])
cbar.set_label("p wave velocity values (m/s)")
axs_t[1,0].set_xticks([])  # Ocultar marcas del eje x
axs_t[1,0].set_xlabel('')  # Ocultar etiqueta del eje x

# tercer gráfico: Density
im3 = axs_t[2,0].imshow(dens2d, cmap="copper", extent=[0, vp2d.shape[1], 0, depth_tot], aspect='auto', vmax=3000)
axs_t[2,0].invert_yaxis()
cbar = fig.colorbar(im3, ax=axs_t[2,0])
cbar.set_label("$Density (kg/m^{3})$")
axs_t[2,0].set_xticks([])  # Ocultar marcas del eje x
axs_t[2,0].set_xlabel('')  # Ocultar etiqueta del eje x

im4 = axs_t[0,1].imshow(SP2d.T, cmap="copper", extent=[0, vp2d.shape[1], 0, depth_tot], aspect='auto')
axs_t[0,1].invert_yaxis()
cbar = fig.colorbar(im4, ax=axs_t[0,1])
cbar.set_label("Sonic porosity (adim)")
axs_t[0,1].set_xticks([])  # Ocultar marcas del eje x
axs_t[0,1].set_xlabel('')  # Ocultar etiqueta del eje x

im7 = axs_t[1,1].imshow(Gr2d.T, cmap="copper", extent=[0, vp2d.shape[1], 0, depth_tot], aspect='auto')
axs_t[1,1].invert_yaxis()
cbar = fig.colorbar(im7, ax=axs_t[1,1])
cbar.set_label("Gamma ray (GAPI)")
axs_t[1,1].set_xticks([])  # Ocultar marcas del eje x
axs_t[1,1].set_xlabel('')  # Ocultar etiqueta del eje x

im6 = axs_t[2,1].imshow(DP2d.T, cmap="copper", extent=[0, DP2d.shape[1], 0, depth_tot], aspect='auto',)
axs_t[2,1].invert_yaxis()
cbar = fig.colorbar(im6, ax=axs_t[2,1])
cbar.set_label("Density porosity (adim)")
axs_t[2,1].set_xticks([])  # Ocultar marcas del eje x
axs_t[2,1].set_xlabel('')  # Ocultar etiqueta del eje x

fig_t.suptitle("2D array for every training Data")
fig_t.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()


# ## SOM

# In[ ]:


#creación del df
columns_df=['Vs','Vp','Density','Sonic Porosity','Gamma ray','Density Porosity']
Depth= depth_tot #index


combined_array_vstack = np.column_stack((vs_smooth, vp_smooth, Dens_kgm3_sm, sonicPor_smooth, GR_sm, DensPor_smooth))
print("Usando np.vstack:\n", combined_array_vstack)

df_training= pd.DataFrame(combined_array_vstack, columns=columns_df, index=depth_m)
df_training


# In[ ]:


data = df_training[df_training.columns[:]]
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

data


# In[ ]:


# Initialization and training
n_neurons = 22
m_neurons = 22
som1 = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som1.pca_weights_init(data)
som1.train(data, 100000, verbose=True)  # random training


# In[ ]:


# Initialization and training
n_neurons = 10
m_neurons = 10
som2 = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som2.pca_weights_init(data)
som2.train(data, 100000, verbose=True)  # random training


# In[ ]:


# Initialization and training
n_neurons = 3
m_neurons = 3
som3 = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som3.pca_weights_init(data)
som3.train(data, 100000, verbose=True)  # random training


# In[ ]:


som1.topographic_error(data[:100])


# In[ ]:


som2.topographic_error(data[:100])


# In[ ]:


som3.topographic_error(data[:100])


# In[ ]:


plt.figure(figsize=(9, 9))

plt.pcolor(som1.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()


# In[ ]:


plt.figure(figsize=(9, 9))

plt.pcolor(som2.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()


# In[ ]:


plt.figure(figsize=(9, 9))

plt.pcolor(som3.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()


# In[ ]:


plt.figure(figsize=(7, 7))
frequencies = som1.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()


# In[ ]:


plt.figure(figsize=(7, 7))
frequencies = som2.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()


# In[ ]:


plt.figure(figsize=(7, 7))
frequencies = som3.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()


# ## calculos para saber #neuronas

# In[ ]:


N=data.shape[0]
num_neuro=5*np.sqrt(N)

dim=int(np.sqrt(num_neuro)+1)
sig=np.sqrt(dim^2 +dim^2)

dim*dim,num_neuro,sig


# In[ ]:


som_n = MiniSom(dim, dim, data.shape[1], sigma=1.5, learning_rate=.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som_n.pca_weights_init(data)
som_n.train(data, 100000, verbose=True)  # random training


# In[ ]:


som_n2 = MiniSom(dim, dim, data.shape[1], sigma=sig, learning_rate=.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som_n2.pca_weights_init(data)
som_n2.train(data, 11500, verbose=True)  # random training


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 9))  # Ajuste del tamaño de la figura

# Primer subplot: Distance map
distance_map = som_n2.distance_map().T  # Transponer el mapa de distancia
im1 = axes[0].pcolor(distance_map, cmap='bone_r')  # Graficar el mapa de distancia
fig.colorbar(im1, ax=axes[0])  # Añadir una barra de color al primer subplot
axes[0].set_title('Distance Map')  # Título del primer subplot

# Segundo subplot: Activation response
frequencies = som_n2.activation_response(data).T  # Transponer las frecuencias
im2 = axes[1].pcolor(frequencies, cmap='Blues')  # Graficar las frecuencias
fig.colorbar(im2, ax=axes[1])  # Añadir una barra de color al segundo subplot
axes[1].set_title('Activation Response')  # Título del segundo subplot

# Mostrar la figura con ambos subplots
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 9))  # Ajuste del tamaño de la figura

# Primer subplot: Distance map
distance_map = som_n.distance_map().T  # Transponer el mapa de distancia
im1 = axes[0].pcolor(distance_map, cmap='bone_r')  # Graficar el mapa de distancia
fig.colorbar(im1, ax=axes[0])  # Añadir una barra de color al primer subplot
axes[0].set_title('Distance Map')  # Título del primer subplot

# Segundo subplot: Activation response
frequencies = som_n.activation_response(data).T  # Transponer las frecuencias
im2 = axes[1].pcolor(frequencies, cmap='Blues')  # Graficar las frecuencias
fig.colorbar(im2, ax=axes[1])  # Añadir una barra de color al segundo subplot
axes[1].set_title('Activation Response')  # Título del segundo subplot

# Mostrar la figura con ambos subplots
plt.show()


# In[ ]:


som_n.pca_weights_init(data)
max_iter = 1000
q_error = []
t_error = []

for i in range(max_iter):
    rand_i = np.random.randint(len(data))
    som_n.update(data[rand_i], som_n.winner(data[rand_i]), i, max_iter)
    q_error.append(som_n.quantization_error(data))
    t_error.append(som_n.topographic_error(data))

plt.plot(np.arange(max_iter), q_error, label='quantization error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('error')
plt.xlabel('iteration index')
plt.legend()
plt.show()

print([q_error,t_error])


# ## Sin el basamento y con solo 10 categorias disponibles :)

# In[ ]:


sig_s10=np.sqrt(2^2 +2^2)


# In[ ]:


som_s10 = MiniSom(2, 5, data.shape[1], sigma=sig_s10, learning_rate=.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som_s10.pca_weights_init(data)
som_s10.train(data, 16500, verbose=True)  # random training


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 9))  # Ajuste del tamaño de la figura

# Primer subplot: Distance map
distance_map = som_s10.distance_map().T  # Transponer el mapa de distancia
im1 = axes[0].pcolor(distance_map, cmap='bone_r')  # Graficar el mapa de distancia
fig.colorbar(im1, ax=axes[0])  # Añadir una barra de color al primer subplot
axes[0].set_title('Distance Map')  # Título del primer subplot

# Segundo subplot: Activation response
frequencies = som_s10.activation_response(data).T  # Transponer las frecuencias
im2 = axes[1].pcolor(frequencies, cmap='Blues')  # Graficar las frecuencias
fig.colorbar(im2, ax=axes[1])  # Añadir una barra de color al segundo subplot
axes[1].set_title('Activation Response')  # Título del segundo subplot

# Mostrar la figura con ambos subplots
plt.show()


# In[ ]:


som_s10.pca_weights_init(data)
max_iter = 1000
q_error = []
t_error = []

for i in range(max_iter):
    rand_i = np.random.randint(len(data))
    som_s10.update(data[rand_i], som_s10.winner(data[rand_i]), i, max_iter)
    q_error.append(som_s10.quantization_error(data))
    t_error.append(som_s10.topographic_error(data))

plt.plot(np.arange(max_iter), q_error, label='quantization error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('error')
plt.xlabel('iteration index')
plt.legend()
plt.show()

print([q_error,t_error])


# In[ ]:


df_training


# In[ ]:


df_training.plot(y='Density Porosity')


# In[ ]:


max_value = df_training['Density Porosity'].max()

# Encontrar el índice del valor máximo
max_index = df_training[df_training['Density Porosity'] == max_value].index

# Eliminar el valor máximo del DataFrame
df_training = df_training.drop(max_index)

# Opcional: Verificar el DataFrame limpio
df_training


# In[ ]:


df_training_cleaned = df_training[(df_training['Density Porosity'] >= -1) & (df_training['Density Porosity'] <= 1) & (df_training['Density'] <= 3000)]


# In[ ]:


df_training_cleaned.plot(y='Density Porosity')


# In[ ]:


df_training_cleaned


# In[ ]:


df_training_cleaned.plot(subplots=True, figsize=(10, 12), layout=(3, 2), sharex=True)
plt.tight_layout()
plt.show()


# In[ ]:


data_f = df_training_cleaned[df_training_cleaned.columns[:]]
# data normalization
data_f = (data_f - np.mean(data_f, axis=0)) / np.std(data_f, axis=0)
data_f = data_f.values


# In[ ]:


N=data_f.shape[0]
num_neuro=5*np.sqrt(N)

dim=int(np.sqrt(num_neuro)+1)
sig=np.sqrt(dim^2 +dim^2)

dim*dim,num_neuro,sig


# In[ ]:


som_filt = MiniSom(dim, dim, data_f.shape[1], sigma=sig, learning_rate=.5, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som_filt.pca_weights_init(data_f)
som_filt.train(data_f, 500*dim*dim, verbose=True)  # random training


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 9))  # Ajuste del tamaño de la figura

# Primer subplot: Distance map
distance_map = som_filt.distance_map().T  # Transponer el mapa de distancia
im1 = axes[0].pcolor(distance_map, cmap='bone_r')  # Graficar el mapa de distancia
fig.colorbar(im1, ax=axes[0])  # Añadir una barra de color al primer subplot
axes[0].set_title('Distance Map')  # Título del primer subplot

# Segundo subplot: Activation response
frequencies = som_filt.activation_response(data_f).T  # Transponer las frecuencias
im2 = axes[1].pcolor(frequencies, cmap='Blues')  # Graficar las frecuencias
fig.colorbar(im2, ax=axes[1])  # Añadir una barra de color al segundo subplot
axes[1].set_title('Activation Response')  # Título del segundo subplot

# Mostrar la figura con ambos subplots
plt.show()


# In[ ]:


som_filt2 = MiniSom(dim, dim, data_f.shape[1], sigma=sig, learning_rate=0.3, 
              neighborhood_function='gaussian', random_seed=0, topology='rectangular')

som_filt2.pca_weights_init(data_f)
som_filt2.train(data_f, 500*dim*dim, verbose=True)  # random training


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 9))  # Ajuste del tamaño de la figura

# Primer subplot: Distance map
distance_map = som_filt2.distance_map().T  # Transponer el mapa de distancia
im1 = axes[0].pcolor(distance_map, cmap='bone_r')  # Graficar el mapa de distancia
fig.colorbar(im1, ax=axes[0])  # Añadir una barra de color al primer subplot
axes[0].set_title('Distance Map')  # Título del primer subplot

# Segundo subplot: Activation response
frequencies = som_filt2.activation_response(data_f).T  # Transponer las frecuencias
im2 = axes[1].pcolor(frequencies, cmap='Blues')  # Graficar las frecuencias
fig.colorbar(im2, ax=axes[1])  # Añadir una barra de color al segundo subplot
axes[1].set_title('Activation Response')  # Título del segundo subplot

fig.suptitle("SOM Distance Map and Activation Response", fontsize=16, y=0.95)

# Ajustar el diseño para que las etiquetas no se superpongan
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar rect para dejar espacio para el título

# Mostrar la figura con ambos subplots
plt.show()


# In[ ]:


(dim)/2


# In[ ]:


def exponential_decay(initial_value, iteration, total_iterations):
    return initial_value * np.exp(-iteration / total_iterations)


# In[ ]:


# Rango de valores para sigma y learning rate
sigma_values = [1,2,3,5.0, 7.5, 10.0, 12.5, 15.0]
learning_rate_values = [0.1, 0.2, 0.3, 0.5, 0.7]

# Definir el número de iteraciones
num_iterations = 500*dim*dim

# Función para la reducción exponencial
def exponential_decay(initial_value, iteration, total_iterations):
    return initial_value * np.exp(-iteration / total_iterations)

best_sigma = None
best_learning_rate = None
best_error = float('inf')

# Grid Search
for sigma in sigma_values:
    for learning_rate in learning_rate_values:
        som = MiniSom(x=22, y=22, input_len=data_f.shape[1], sigma=sigma, learning_rate=learning_rate)
        som.random_weights_init(data_f)
        
        initial_sigma = sigma
        initial_learning_rate = learning_rate
        
        for i in range(num_iterations):
            som.train_random(data_f, num_iteration=1)
            # Ajustar sigma y learning rate dinámicamente
            current_sigma = exponential_decay(initial_sigma, i, num_iterations)
            current_learning_rate = exponential_decay(initial_learning_rate, i, num_iterations)
            som.sigma = current_sigma
            som.learning_rate = current_learning_rate
        
        quantization_error = np.mean([np.linalg.norm(d - som.get_weights()[som.winner(d)]) for d in data_f])
        
        if quantization_error < best_error:
            best_error = quantization_error
            best_sigma = sigma
            best_learning_rate = learning_rate

print(f"Best sigma: {best_sigma}, Best learning rate: {best_learning_rate}, Quantization error: {best_error}")


# In[ ]:


# Obtener las coordenadas del BMU para cada dato
bmu_coordinates = np.array([som.winner(x) for x in data_f])

# Asignar un color único a cada BMU
bmu_colors = {bmu: plt.cm.jet(i / (22 * 22)) for i, bmu in enumerate(set(tuple(bmu) for bmu in bmu_coordinates))}

# Crear un scatter plot de los datos, coloreados por su BMU
plt.figure(figsize=(10, 8))
for i, x in enumerate(data_f):
    bmu = tuple(bmu_coordinates[i])
    plt.scatter(x[0], x[1], color=bmu_colors[bmu], s=10)  # s=10 ajusta el tamaño de los puntos

plt.title('Datos clasificados por el SOM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[ ]:


bmu_indices = bmu_coordinates[:, 0] * som.get_weights().shape[1] + bmu_coordinates[:, 1]

df_training_cleaned['SOM_Classification'] = bmu_indices
df_training_cleaned


# In[ ]:


df_training_cleaned.plot(y='SOM_Classification')
clas=df_training_cleaned['SOM_Classification']


# In[ ]:


df_training_cleaned['SOM_Classification'] = bmu_indices
df_training_cleaned


# In[ ]:


#clas
Nx = 10000
Nz_cl = int(clas.size)
cl2d = np.zeros([Nx,Nz_cl])
for i in range(0,Nx):
    cl2d[i] = clas


# In[ ]:


colors = ['#4b3b42', '#9c8481', '#d9c2ba','#e2cbb0','#ffd592']
cmapd = mcolors.ListedColormap(colors)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))  

# Supongamos que `cl2d` es tu matriz de datos
# Primer gráfico
im3 = ax.imshow(cl2d.T, cmap=cmapd)
cbar = fig.colorbar(im3, ax=ax)  
cbar.set_label("Classification of lithofacies using SOM")

plt.show()


# In[ ]:


colors = ['#4b3b42', '#9c8481', '#d9c2ba','#ffd592']
cmapd = mcolors.ListedColormap(colors)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))  

# Supongamos que `cl2d` es tu matriz de datos
# Primer gráfico
im3 = ax.imshow(cl2d.T, cmap=cmapd)
cbar = fig.colorbar(im3, ax=ax)  
cbar.set_label("Classification of lithofacies using SOM")

plt.show()


# In[ ]:


depth_m=df_training_cleaned.index

num_pixels = cl2d.shape[0]
depth_tot = max(depth_m)

print(str(depth_tot)+' m is the maximum depth')


# In[ ]:


depth_per_pixel = depth_tot / num_pixels
depth_per_pixel


# In[ ]:


depths = np.arange(0, depth_tot, depth_per_pixel)
depths


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(10, 10))  

im3 = ax.imshow(cl2d.T, cmap=cmapd, extent=[0, vp2d.shape[1], depth_tot, 0], aspect='auto')  # Invertir el eje y
cbar = fig.colorbar(im3, ax=ax)  
cbar.set_label("Classification of lithofacies using SOM")

# Eliminar las etiquetas del eje x
ax.set_xticks([])
ax.set_xlabel('')

ax.set_ylabel('Depth(m)')

# Mostrar la figura
plt.show()


# In[ ]:




