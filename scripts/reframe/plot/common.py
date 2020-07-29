import numpy as np
import matplotlib
# import matplotlib_terminal
import matplotlib.pyplot as plt
import re

# {{{ def plot_data
def plot_data(myplt_data, rows_labels, columns_labels, dim_to_plot):
    fig, ax = plt.subplots()
    mylabel = "Average Execution Time (s/iteration)"
    mypng = 'heatmap_matplot_sqpatch_elapsed_' + dim_to_plot + '.png'
    # mylabel = "Metrics [count|erg]")
    im, cbar = heatmap(myplt_data, rows_labels, columns_labels, ax=ax,
                       cmap="YlGn", cbarlabel=mylabel)
    texts = annotate_heatmap(im, valfmt="{x:.2f} s")
    fig.tight_layout()
    #plt.show()
    plt.savefig(mypng)
    plt.close()
    print(mypng, 'written')
# }}}

# {{{ def shape_data
def shape_data(rows, dim_to_plot, dim_first_key, prg_model_first_key):
    list_of_lists = []
    for prg_env in rows[dim_first_key][prg_model_first_key].keys():
        elapsed_l = []
        for prg_model in rows[dim_first_key].keys():
            elapsed_l.append(rows[dim_to_plot][prg_model][prg_env]['avg_elapsed'])
            # elapsed_l.append(rows[dim_first_key][prg_model][prg_env]['avg_elapsed'])
        list_of_lists.append(elapsed_l)

    myplt_data = np.array(list_of_lists)
    return myplt_data
# }}}

# {{{ def update_dict
def update_dict(json_dict, rows, cubeside_d):
    for ii in range(len(json_dict['runs'][0]['testcases'])):
        dd = json_dict['runs'][0]['testcases'][ii]
        ll = len(dd['perfvars'])
        #print(prgenv_, prgmodel_)
        for lll in range(ll):
            name_ = dd['perfvars'][lll]['name']
            #value_ = d['runs'][0]['testcases'][0]['perfvars'][lll]['value']
            #unit_ = d['runs'][0]['testcases'][0]['perfvars'][lll]['unit']    
            if name_ == 'Elapsed':
                #elapsed_ = d['runs'][0]['testcases'][0]['perfvars'][lll]['value']
                elapsed_ = dd['perfvars'][lll]['value']
            if name_ == 'mpi_ranks':
                mpi_ = dd['perfvars'][lll]['value']
            if name_ == 'cubeside':
                cubeside_ = dd['perfvars'][lll]['value']
            if name_ == 'steps':
                steps_ = dd['perfvars'][lll]['value']
                avg_elapsed_ = elapsed_ / ( steps_ + 1 )
            if name_ == 'Total Neighbors':
                total_neighb_ = dd['perfvars'][lll]['value']
            if name_ == 'Avg neighbor count per particle':
                avg_neighb_ = dd['perfvars'][lll]['value']
            if name_ == 'Total energy':
                total_energy = dd['perfvars'][lll]['value']
            if name_ == 'Internal energy':
                int_energy = dd['perfvars'][lll]['value']
                dim = cubeside_d[mpi_]
                prg_model = re.sub(r'\d+','', dd['name'].replace('SphExa_', '').replace('_Check', ''))[:-1]
                prg_env = dd['environment']
                # prg_env = dd['environment'].replace('PrgEnv-', '')
                # title_ = cubeside_d[mpi_] + '-' + prgmodel_.split('_')[1] + '-' + prgenv_.replace('PrgEnv-', '')
                # row[title_] = [total_neighb_, avg_neighb_, total_energy, int_energy]
                # rows[title_] = [avg_elapsed_]
                rows[dim][prg_model][prg_env]['mpi'] = mpi_
                rows[dim][prg_model][prg_env]['steps'] = steps_
                rows[dim][prg_model][prg_env]['cubeside'] = cubeside_
                rows[dim][prg_model][prg_env]['elapsed'] = elapsed_
                rows[dim][prg_model][prg_env]['avg_elapsed'] = avg_elapsed_
                rows[dim][prg_model][prg_env]['total_neighb'] = total_neighb_
                rows[dim][prg_model][prg_env]['avg_neighb'] = avg_neighb_
                rows[dim][prg_model][prg_env]['total_energy'] = total_energy
                rows[dim][prg_model][prg_env]['int_energy'] = int_energy
                # print(title_,
                #       avg_elapsed_,
                #       total_neighb_, avg_neighb_,
                #       total_energy, int_energy)

    return rows
# }}}

# {{{ def init_dict
def init_dict(dims, prg_models, prg_envs):
    """
    Initialise rows[dim][prg_model][prg_env] dict with -1
    """
    # print('dims=', dims)
    # print('prg_models=', prg_models)
    # print('prg_envs=', prg_envs)
    # d['small']['MPI_OpenMP_Target']['gnu']['mpi'] = 999
    # d['small']['MPI_OpenMP_Target']['gnu']['cubeside'] = 999
    rows = {}
    for dim in dims:
        rows[dim] = {}
        for prg_model_ in prg_models:
            prg_model = prg_model_.replace('SphExa_', '').replace('_Check', '')
            rows[dim][prg_model] = {}
            for prg_env_ in prg_envs:
                prg_env = prg_env_
                # prg_env = prg_env_.replace('PrgEnv-', '')
                rows[dim][prg_model][prg_env] = {
                    'mpi': -1,
                    'steps': -1,
                    'cubeside': -1,
                    'elapsed': -1,
                    'avg_elapsed': -1,
                    'total_neighb': -1,
                    'avg_neighb': -1,
                    'total_energy': -1,
                    'int_energy': -1,
                    }

    # print(rows)
    return rows

# print(row)
# print(row['small']['MPI_OpenMP_Target']['gnu'])
# init_dict(row)
# print(row['small']['MPI_OpenMP_Target']['gnu'])
# }}}

# {{{ def heatmap
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
# }}}

# {{{ def annotate_heatmap
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
# }}}
