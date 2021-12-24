import numpy as onp
import matplotlib.pyplot as plt


# =============================================================
# showmat
# =============================================================
def showmat(X, xlabel=None, ylabel=None, fontsize=14, crange=None, figsize=None, xticklabel=None, yticklabel=None, cmap=None):
    """Show 2D ndarray as matrix.
    Args:
        x: 2D ndarray
        xlabel: (option) x-axis label
        ylabel: (option) y-axis label
        fontsize: (option) font size
        crange: (option) colormap range, [min, max] or "maxabs"
        figsize: (option) figure size
    """

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [1, 1]
    X = X.copy()
    if len(X.shape) > 2:
        print("X has to be matrix or vector")
        return

    if X.shape[0]==1 or X.shape[1]==1:
        Nsize = X.size
        X = X.reshape(onp.sqrt(Nsize), onp.sqrt(Nsize))

    # Plot ----------------------------------------------------
    fig = plt.figure(figsize=(8*figsize[0], 6*figsize[1]))

    if cmap is None:
        plt.imshow(X,interpolation='none',aspect='auto')
    else:
        plt.imshow(X, interpolation='none', aspect='auto', cmap=cmap)
    plt.colorbar()

    # Color range
    if not(crange is None):
        if len(crange)==2:
            plt.clim(crange[0], crange[1])

        elif crange == "maxabs":
            xmaxabs = onp.absolute(X).max()
            plt.clim(-xmaxabs, xmaxabs)

    if not(xlabel is None):
        plt.xlabel(xlabel)
    if not(ylabel is None):
        plt.ylabel(ylabel)
    if xticklabel is not None:
        plt.gca().set_xticks(onp.arange(0,X.shape[1]))
        plt.gca().set_xticklabels(xticklabel)
    if yticklabel is not None:
        plt.gca().set_yticks(onp.arange(0,X.shape[0]))
        plt.gca().set_yticklabels(yticklabel)
        # ax.set_xticklabels(xticklabel)

    plt.rcParams["font.size"] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)


# =============================================================
# showtimedata
# =============================================================
def showtimedata(X, xlabel="Time", ylabel="Channel", fontsize=14, linewidth=1.5,
                 intervalstd=10, figsize=None):
    """Show 2D ndarray as time series
    Args:
        x: signals. 2D ndarray [num_channel, num_time]
        xlabel: (option) x-axis label
        ylabel: (option) y-axis label
        fontsize: (option) font size
        linewidth: (option) width of lines
        intervalstd: (option) interval between lines based on maximum std.
        figsize: (option) figure size
    """

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [2, 1]
    X = X.copy()
    X = X.reshape([X.shape[0],-1])

    if X.shape[1]==1:
        X = X.reshape([1,-1])

    Nch = X.shape[0]
    Nt = X.shape[1]

    vInterval = X.std(axis=1).max() * intervalstd
    vPos = vInterval * (onp.arange(Nch,0,-1) - 1)
    vPos = vPos.reshape([1, -1]).T  # convert to column vector
    X = X + vPos

    # Plot ----------------------------------------------------
    fig = plt.figure(figsize=(8*figsize[0], 6*figsize[1]))

    for i in range(Nch):
        plt.plot(list(range(Nt)), X[i,:], linewidth=linewidth)

    plt.xlim(0, Nt-1)
    plt.ylim(X.min(),X.max())

    ylabels = [str(num) for num in range(Nch)]
    plt.yticks(vPos,ylabels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.rcParams["font.size"] = fontsize

    plt.ion()
    plt.show()
    plt.pause(0.001)


# =============================================================
# showhist
# =============================================================
def showhist(X, bins=100, xlabel=None, ylabel=None, fontsize=14, crange=None, figsize=None):
    """Show 2D ndarray as matrix.
    Args:
        x: 2D ndarray
        xlabel: (option) x-axis label
        ylabel: (option) y-axis label
        fontsize: (option) font size
        crange: (option) colormap range, [min, max] or "maxabs"
        figsize: (option) figure size
    """

    # Prepare plot data ---------------------------------------
    if figsize is None:
        figsize = [1, 1]
    X = X.copy()
    if len(X.shape) > 2:
        print("X has to be matrix or vector")
        return

    fig = plt.figure(figsize=(8*figsize[0], 6*figsize[1]))
    plt.hist(X, bins=bins)