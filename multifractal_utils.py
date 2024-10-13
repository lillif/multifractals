import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import matplotlib.axes as mpl_axes

def plot_moments(moments: np.ndarray,
                 R: np.ndarray,
                 Q: np.ndarray,
                 fig,
                 ax: mpl_axes.Axes,
                 fitting_range: tuple[int, int],
                 pixel_reso: float = 0.05,
                 normalise: bool = True,
                 colours = sns.color_palette("flare", n_colors=10)): # : sns.palettes._ColorPalette 
    
    R = R * pixel_reso * 111 # 0.05 degrees, 1 degree approx = 111km, 1km = 1e3m
    
    axs = [ax for i in range(10)]

    for q, ax, col in zip(Q, axs, colours):
        if normalise:
            normalised_moment_q = moments[:, q-1] / moments[0, q-1]
            ax.plot(R, normalised_moment_q, color=col, label=f'q = {q:.0f}')
        else:
            ax.plot(R, moments[:, q-1], color=col, label=f'q = {q:.0f}')
 
    ax.set_xscale('log')
    ax.set_yscale('log')

    if normalise:
        ax.set_ylabel(r'$\hat{S}_{q, norm}(r)$')
    else:
        ax.set_ylabel(r'$\hat{S}_{q}(r)$')
        
    ylims = ax.get_ylim()

    ax.fill_between(R, 0, ylims[1], 
                    where = (R >= fitting_range[0] * pixel_reso * 111 ) & (R <= fitting_range[1] * pixel_reso * 111),
                    facecolor='red', alpha=0.1, label='fitting range')

    ax.set_xlabel('r (km)')

    cbar_ax_q = inset_axes(ax, width="2%", height="95%", loc='right', 
                       bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)


    
    # color palette p to colormap
    q_cmap = cm.colors.ListedColormap(colours)
    
    # define norm for colorbar between 1 and 10
    norm = cm.colors.Normalize(vmin=1, vmax=10)
    cbar = plt.colorbar(cm.ScalarMappable(cmap=q_cmap, norm=norm), cax=cbar_ax_q)
    cbar.set_label(r'$q$', fontsize=9.5)
    cbar.set_ticks(np.arange(1,11))
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_ylim(ylims)




def plot_diurnal_bootstrap(zetas: dict,
                           zeta_bounds: dict,
                           ax: mpl_axes.Axes,
                           color,
                           label: str,
                           conf: int,
                           GMT_delta: int):

    hours = list(zetas.keys())
    hours.sort()
    
        
    # convert to local time
    lower_zeta_bounds = [zeta_bounds[hour][0] for hour in hours]
    upper_zeta_bounds = [zeta_bounds[hour][1] for hour in hours]

    zetas_list = [zetas[hour] for hour in hours]
    
    local_time_zetas = zetas_list[-1*GMT_delta:] + zetas_list[:-1*GMT_delta]
    
    local_time_lower_zeta_bounds = lower_zeta_bounds[-1*GMT_delta:] + lower_zeta_bounds[:-1*GMT_delta]
    local_time_upper_zeta_bounds = upper_zeta_bounds[-1*GMT_delta:] + upper_zeta_bounds[:-1*GMT_delta]
    
    # plot
    line = ax.plot(local_time_zetas, '-x', markersize=4, markeredgewidth=1.5, color=color, linewidth=1, label=label)
    
    ax.fill_between(np.arange(24),
                    local_time_lower_zeta_bounds,
                    local_time_upper_zeta_bounds,
                    alpha=0.2,
                    edgecolor=color,
                    facecolor=color)
    
    ax.set_xlabel(f'Local time (GMT{GMT_delta:.0f})', fontsize=10)
    ax.set_xlim([0,23])
    
    return line
