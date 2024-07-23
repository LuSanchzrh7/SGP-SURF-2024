#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lasio
from scipy import signal
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


# In[3]:


sns.set(style="whitegrid")


# In[4]:


las= lasio.read('Sonic_P_and_S_Output_7240ft_350ft.las')


# In[5]:


las.sections.keys()


# In[6]:


las.sections['Curves']


# In[7]:


for curve in las.curves:
    print(curve.mnemonic)


# In[8]:


for count, curve in enumerate(las.curves):
    print(f"Curve: {curve.mnemonic}, \t Units: {curve.unit}, \t Description: {curve.descr}")
print(f"There are a total of: {count+1} curves present within this file")


# In[9]:


well = las.df()


# In[10]:


well.head()


# In[11]:


well.index


# In[12]:


well.plot(y='DTCO')


# In[13]:


well.plot(y='DTSM')


# In[14]:


well.plot(y='VPVS')


# In[15]:


well['Vs']=1/well['DTSM']
well.plot(y='Vs',ylabel='velocity(ft/us)')


# In[16]:


well['Vp']=1/well['DTCO']
well.plot(y='Vp',ylabel='velocity(ft/us)')


# In[17]:


fig=plt.figure(figsize=(10,20))
ax1= fig.add_subplot(1,2,1)
ax2= fig.add_subplot(1,2,2)
ax1.plot(well['Vs'],well.index, color='r')
ax2.plot(well['Vp'],well.index, color='g')
ax1.invert_yaxis()
ax2.invert_yaxis()

ax1.set_xlabel('Velocity (ft/$\mu$s)')
ax1.set_ylabel('Depth (ft)')
ax2.set_xlabel('Velocity (ft/$\mu$s)')
ax2.set_ylabel('Depth (ft)')

ax1.set_title('Vs')
ax2.set_title('Vp')


# In[18]:


plt.figure(figsize=(8, 20))

plt.plot(well['VPVS'],well.index,'b-',label='Vp/Vs')
plt.ylabel('DEPTH (ft)')
plt.title("Vp/Vs")
plt.legend()

plt.gca().invert_yaxis()

plt.show()


# In[19]:


fig1=plt.figure(figsize=(10,20))
ax11= fig1.add_subplot(1,2,1)
ax12= fig1.add_subplot(1,2,2)
ax11.plot(1/well['DTSM_FAST'],well.index, color='m')
ax12.plot(1/well['DTST'],well.index, color='orange')
ax11.invert_yaxis()
ax12.invert_yaxis()

ax11.set_xlabel('Velocity fast shear waves (ft/$\mu$s)')
ax11.set_ylabel('Depth (ft)')
ax12.set_xlabel('Velocity Stoneley waves(ft/$\mu$s)')
ax12.set_ylabel('Depth (ft)')

ax11.set_title('fast shear waves')
ax12.set_title('Stoneley waves')


# In[20]:


# Crear la figura con una cuadrícula de 1 fila y 4 columnas
fig, axes = plt.subplots(1, 4, figsize=(20, 10))

marker_size=3

# Primer gráfico (Vs)
axes[0].plot(well['Vs'], well.index, color='r', marker='o', linestyle='None', markersize=marker_size)
axes[0].invert_yaxis()
axes[0].set_xlabel('Velocity (ft/$\mu$s)')
axes[0].set_ylabel('Depth (ft)')
axes[0].set_title('Vs')

# Segundo gráfico (Vp)
axes[1].plot(well['Vp'], well.index, color='g', marker='o', linestyle='None', markersize=marker_size)
axes[1].invert_yaxis()
axes[1].set_xlabel('Velocity (ft/$\mu$s)')
axes[1].set_ylabel('Depth (ft)')
axes[1].set_title('Vp')

# Tercer gráfico (DTSM_FAST)
axes[2].plot(1/well['DTSM_FAST'], well.index, color='m', marker='o', linestyle='None', markersize=marker_size)
axes[2].invert_yaxis()
axes[2].set_xlabel('Velocity fast shear waves (ft/$\mu$s)')
axes[2].set_ylabel('Depth (ft)')
axes[2].set_title('Fast shear waves')

# Cuarto gráfico (DTST)
axes[3].plot(1/well['DTST'], well.index, color='orange', marker='o', linestyle='None', markersize=marker_size)
axes[3].invert_yaxis()
axes[3].set_xlabel('Velocity Stoneley waves (ft/$\mu$s)')
axes[3].set_ylabel('Depth (ft)')
axes[3].set_title('Stoneley waves')

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Mostrar la figura
plt.show()


# In[21]:


fig_n, axes_n = plt.subplots(1, 2, figsize=(20, 15))

# Primer gráfico (Vp/Vs)
axes_n[0].plot(well['VPVS'], well.index, color='darkorchid', marker='o', linestyle='None', markersize=marker_size)
axes_n[0].invert_yaxis()
axes_n[0].set_xlabel('ratio vp/vs')
axes_n[0].set_ylabel('Depth (ft)')
axes_n[0].set_title('Vs')

axes_n[1].plot(well['VPVS_FAST'], well.index, color='hotpink', marker='o', linestyle='None', markersize=marker_size)
axes_n[1].invert_yaxis()
axes_n[1].set_xlabel('ratio $vp/vs_{fast}$')
axes_n[1].set_ylabel('Depth (ft)')
axes_n[1].set_title('Vp/Vs_FAST')

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Mostrar la figura
plt.show()


# ### Smooth the data

# In[22]:


well_clean = well.dropna(subset=['Vs', 'Vp', 'DTSM_FAST', 'DTST','VPVS','VPVS_FAST','RHOZ'])

depth = well_clean.index.values


# In[23]:


def moving_average(data, window_size):
    averaged_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            averaged_data.append(np.nan)  # Si no hay suficientes datos para llenar la ventana, agregar NaN
        else:
            window = data[i - window_size + 1:i + 1]
            window_average = np.mean(window)
            averaged_data.append(window_average)
    return np.array(averaged_data)


# In[24]:


filt_l=200
tair_mov_average=moving_average(well_clean['Vs'].values,filt_l)


# In[25]:


plt.figure(figsize=(4,7))

plt.plot(well_clean['Vs'],depth)
plt.plot(tair_mov_average,depth)
plt.plot(signal.savgol_filter(well_clean['Vs'].values, 250, 5),depth)
plt.gca().invert_yaxis()


# In[26]:


print(well['Vs'].values)


# In[27]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=well_clean['Vs'].values,
    y=depth,
    mode='markers',
    marker=dict(size=2, color='mediumpurple'),
    name='Vs'
))

fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['Vs'].values, 250, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=6,
        color='black',
        symbol='triangle-up'
    ),
    name='Savitzky-Golay pol 3'
))
fig.add_trace(go.Scatter(
    x=tair_mov_average,  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=6,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='mov average'
))
fig.update_layout(
    width=800,  # Ajusta el ancho según tus necesidades
    height=1000,  # Ajusta la altura para hacer la figura más larga
    yaxis_autorange='reversed' 
)
fig.show()


# In[28]:


#todas las graficas lol


# In[29]:


fig = make_subplots(rows=1, cols=4, shared_yaxes=True, subplot_titles=("Vs", "Vp", "Fast shear waves", "Stoneley waves"))

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['Vs'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='mediumpurple',
        symbol='triangle-up'
    ),
    name='Vs'
),row=1, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['Vp'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='mediumaquamarine',
        symbol='triangle-up'
    ),
    name='Vp'
),row=1, col=2)

# Añadir la tercera subtrama (Fast shear waves)

fig.add_trace(go.Scatter(
    x=signal.savgol_filter(1/well_clean['DTSM_FAST'].values, 150, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Fast shear waves'
),row=1, col=3)

# Añadir la cuarta subtrama (Stoneley waves)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(1/well_clean['DTST'].values, 150, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='steelblue',
        symbol='triangle-up'
    ),
    name='Stoneley waves'
),row=1, col=4)

# Invertir el eje y en todas las subtramas
fig.update_yaxes(autorange='reversed')

# Ajustar el diseño
fig.update_layout(
    width=1200,  # Ajusta el ancho de la figura
    height=600,  # Ajusta la altura de la figura
    showlegend=False,
    title_text="Well Data"
)

# Mostrar la figura
fig.show()


# In[30]:


well.plot(y='RHOZ')


# In[31]:


fig = make_subplots(rows=1, cols=7, shared_yaxes=True, subplot_titles=("Density", "Vs", "Vp", "Fast shear waves", "Stoneley waves","Vp/Vs", "Vp/vs_Fast"))

fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['RHOZ'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Density'
),row=1, col=1)

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['Vs'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='mediumpurple',
        symbol='triangle-up'
    ),
    name='Vs'
),row=1, col=2)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['Vp'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='mediumaquamarine',
        symbol='triangle-up'
    ),
    name='Vp'
),row=1, col=3)

# Añadir la tercera subtrama (Fast shear waves)

fig.add_trace(go.Scatter(
    x=signal.savgol_filter(1/well_clean['DTSM_FAST'].values, 150, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Fast shear waves'
),row=1, col=4)

# Añadir la cuarta subtrama (Stoneley waves)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(1/well_clean['DTST'].values, 150, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='steelblue',
        symbol='triangle-up'
    ),
    name='Stoneley waves'
),row=1, col=5)

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['VPVS'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='mediumpurple',
        symbol='triangle-up'
    ),
    name='Vp/Vs'
),row=1, col=6)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=signal.savgol_filter(well_clean['VPVS_FAST'].values, 300, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    y=depth,
    marker=dict(
        size=2,
        color='mediumaquamarine',
        symbol='triangle-up'
    ),
    name='Vp'
),row=1, col=7)

# Añadir la tercera subtrama (Fast shear waves)



# Invertir el eje y en todas las subtramas
fig.update_yaxes(autorange='reversed')

# Ajustar el diseño
fig.update_layout(
    width=1200,  # Ajusta el ancho de la figura
    height=600,  # Ajusta la altura de la figura
    showlegend=False,
    title_text="Well Data"
)

# Mostrar la figura
fig.show()


# In[32]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=("Density", "Vs", "Vp"))

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['RHOZ'].values,
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Density'
),row=1, col=1)

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['Vp'].values,
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='Vp'
),row=2, col=1)

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['Vs'].values,
    marker=dict(
        size=2,
        color='orange',
        symbol='triangle-up'
    ),
    name='Vs'
),row=3, col=1)
fig.update_xaxes(title_text="Depth", row=3, col=1)


# ##### Datos sin outliers

# In[33]:


sns.boxplot(well_clean['Vp'].values)


# In[34]:


sns.boxplot(well_clean['Vs'].values)


# In[35]:


def removal_box_plot(df, column, threshold,threshold2):
    sns.boxplot(df[column].values)
    plt.title(f'Original Box Plot of {column}')
    plt.show()
 
    removed_outliers1 = df[df[column].values <= threshold]
    removed_outliers = removed_outliers1[removed_outliers1[column].values >= threshold2]
 
    sns.boxplot(removed_outliers[column].values)
    plt.title(f'Box Plot without Outliers of {column}')
    plt.show()
    return removed_outliers


# In[36]:


threshold_value = 2.95
threshold_value2 =2.3
 
no_outliers = removal_box_plot(well_clean, 'RHOZ', threshold_value,threshold_value2)


# In[37]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=("Density", "Vs", "Vp"))

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=no_outliers['RHOZ'].values,
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Density'
),row=1, col=1)

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['Vp'].values,
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='Vp'
),row=2, col=1)

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['Vs'].values,
    marker=dict(
        size=2,
        color='orange',
        symbol='triangle-up'
    ),
    name='Vs'
),row=3, col=1)
fig.update_xaxes(title_text="Depth", row=3, col=1)


# Hay que suavizar los datos y ponerlos hasta 2000 ft por lo que es para la subsurface level

# In[38]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=("Density", "Vs", "Vp"))

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=no_outliers['RHOZ'].values,
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Density'
),row=1, col=1)

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['Vp'].values,
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='Vp'
),row=2, col=1)

fig.add_trace(go.Scatter(
    x=depth,  # Ventana de tamaño 53 y polinomio de orden 3
    y=well_clean['Vs'].values,
    marker=dict(
        size=2,
        color='orange',
        symbol='triangle-up'
    ),
    name='Vs'
),row=3, col=1)
fig.update_xaxes(title_text="Depth", range=[946, 2000], row=1, col=1)
fig.update_xaxes(title_text="Depth", range=[395, 2000], row=2, col=1)
fig.update_xaxes(title_text="Depth", range=[395, 2000], row=3, col=1)


# In[39]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=("Density", "Vs", "Vp"))

fig.add_trace(go.Scatter(
    x=depth,
    y=signal.savgol_filter(well_clean['RHOZ'].values, 56, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='purple',
        symbol='triangle-up'
    ),
    name='Density'
),row=1, col=1)

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth,
    y=signal.savgol_filter(well_clean['Vs'].values, 50, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='mediumpurple',
        symbol='triangle-up'
    ),
    name='Vs'
),row=2, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=depth,
    y=signal.savgol_filter(well_clean['Vp'].values, 50, 5),  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='Vp'
),row=3, col=1)



# Ajustar el diseño
fig.update_xaxes(range=[395, 2400], row=1, col=1)
fig.update_xaxes(range=[395, 2400], row=2, col=1)
fig.update_xaxes(title_text="Depth", range=[395, 2400], row=3, col=1)
# Mostrar la figura
fig.show()


# In[40]:


Density_smooth = signal.savgol_filter(well_clean['RHOZ'].values, 200, 3)
Vp_smooth = signal.savgol_filter(well_clean['Vp'].values, 200, 3)
Vs_smooth = signal.savgol_filter(well_clean['Vs'].values, 200, 3)


# In[41]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=("Density", "Vs", "Vp"))

fig.add_trace(go.Scatter(
    x=depth,
    y=well_clean['RHOZ'].values,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening'
),row=1, col=1)
fig.add_trace(go.Scatter(
    x=depth,
    y=Density_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening'
),row=1, col=1)

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth,
    y=well_clean['Vs'].values,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
),row=2, col=1)
fig.add_trace(go.Scatter(
    x=depth,
    y=Vs_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)

# Añadir la segunda subtrama (Vp)
fig.add_trace(go.Scatter(
    x=depth,
    y=well_clean['Vp'].values,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)
fig.add_trace(go.Scatter(
    x=depth,
    y=Vp_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)



# Ajustar el diseño
fig.update_xaxes(range=[395, 5600], row=1, col=1)
fig.update_xaxes(range=[395, 5600], row=2, col=1)
fig.update_xaxes(title_text="Depth", range=[395, 5400], row=3, col=1)
# Mostrar la figura
fig.show()


# In[42]:


Nx = 10000
Nz = int(well_clean['RHOZ'].size)
den2d = np.zeros([Nx,Nz])
for i in range(0,Nx):
    den2d[i] = Density_smooth
    


# In[43]:


plt.imshow(den2d.T)


# In[44]:


Nx = 10000
Nz = int(well_clean['Vs'].size)
vs2d = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vs2d[i] = Vs_smooth
    


# In[45]:


plt.imshow(vs2d.T)


# In[46]:


Nx = 10000
Nz = int(well_clean['Vp'].size)
vp2d = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vp2d[i] = Vp_smooth
    


# In[47]:


plt.imshow(vp2d.T)


# ## Conv de unidades

# In[48]:


#unidades ft/us a m/s
vp_ms=Vp_smooth*0.3048*(10**6)
vs_ms=Vs_smooth*0.3048*(10**6)

vp_org_ms= well_clean['Vp']*0.3048*(10**6)
vs_org_ms= well_clean['Vs']*0.3048*(10**6)

depth_m=depth*0.3048


# In[49]:


fig = make_subplots(rows=3, cols=1, shared_yaxes=True, subplot_titles=("Vp", "Vs", "density"))

# Añadir la primera subtrama (Vs)
fig.add_trace(go.Scatter(
    x=depth_m,
    y=vs_org_ms,  # Ventana de tamaño 53 y polinomio de orden 3
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
    y=vs_ms,  # Ventana de tamaño 53 y polinomio de orden 3
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
    y=vp_org_ms,  # Ventana de tamaño 53 y polinomio de orden 3
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
    y=vp_ms,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=2, col=1)


fig.add_trace(go.Scatter(
    x=depth_m,
    y=well_clean['RHOZ'].values,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='blue',
        symbol='triangle-up'
    ),
    name='Before smoothening',
    showlegend=False
),row=3, col=1)

fig.add_trace(go.Scatter(
    x=depth_m,
    y=Density_smooth,  # Ventana de tamaño 53 y polinomio de orden 3
    marker=dict(
        size=2,
        color='hotpink',
        symbol='triangle-up'
    ),
    name='After smoothening',
    showlegend=False
),row=3, col=1)


# Ajustar el diseño
#fig.update_xaxes(range=[395, 5600], row=1, col=1)
#fig.update_xaxes(range=[395, 5600], row=2, col=1)
fig.update_xaxes(title_text="Depth (m)", row=3, col=1)
fig.update_yaxes(title_text="Vp (m/s)", row=1, col=1)
fig.update_yaxes(title_text="Vs (m/s)", row=2, col=1)
fig.update_yaxes(title_text="$Density (g/cm^{3})$", row=3, col=1)
# Mostrar la figura
fig.show()


# In[50]:


Nx = 10000
Nz = int(vp_ms.size)
vp2d = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vp2d[i] = vp_ms


# In[51]:


im = plt.imshow(vp2d.T, cmap="copper")
cbar = plt.colorbar(im)
cbar.set_label("Colorbar")
plt.show()


# In[52]:


Nx = 10000
Nz = int(vs_ms.size)
vs2d = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vs2d[i] = vs_ms


# In[54]:


im2 = plt.imshow(vs2d.T, cmap="copper")
cbar = plt.colorbar(im2)
cbar.set_label("velocity values (m/s)")
plt.show()


# In[55]:


VS_250=signal.savgol_filter(well_clean['Vs'].values, 200, 5)*0.3048*(10**6)
Nx = 10000
Nz = int(VS_250.size)
vs2d250 = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vs2d250[i] = VS_250


# In[56]:


im3 = plt.imshow(vs2d250.T, cmap="copper")
cbar = plt.colorbar(im3)
cbar.set_label("velocity values (m/s)")
plt.show()


# In[57]:


fig, axs = plt.subplots(1, 2, figsize=(10, 5))  

# Primer gráfico 250,5
im3 = axs[0].imshow(vs2d250.T, cmap="copper")
cbar = fig.colorbar(im3, ax=axs[0])  
cbar.set_label("velocity values (m/s)")

# Segundo gráfico 250,3
im2 = axs[1].imshow(vs2d.T, cmap="copper")
cbar = fig.colorbar(im2, ax=axs[1])
cbar.set_label("velocity values (m/s)")

plt.show()


# ### función de smoothening manual

# In[58]:


velp_prom_ms=moving_average(vp_org_ms.values,filt_l)
vels_prom_ms=moving_average(vs_org_ms.values,filt_l)


# In[59]:


Nx = 10000
Nz = int(velp_prom_ms.size)
vp_pr = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vp_pr[i] = velp_prom_ms


# In[60]:


im3n = plt.imshow(vp_pr.T, cmap="copper")
cbar = plt.colorbar(im3n)
cbar.set_label("velocity values (m/s)")
plt.show()


# In[61]:


Nx = 10000
Nz = int(vels_prom_ms.size)
vs_pr = np.zeros([Nx,Nz])
for i in range(0,Nx):
    vs_pr[i] = vels_prom_ms


# In[62]:


im2n = plt.imshow(vs_pr.T, cmap="copper")
cbar = plt.colorbar(im2n)
cbar.set_label("velocity values (m/s)")
plt.show()

