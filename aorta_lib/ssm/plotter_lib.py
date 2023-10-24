import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
#mpl.use('pdf')

class Plotter():
    def __init__(self, scale='log', plot_period=10):
        assert plot_period > 0, f"plot_period is {plot_period}"
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('iter')
        ax.set_ylabel('loss')
        if scale: ax.set_yscale(scale)
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)

        self.ax = ax
        self.fig = fig
        self.plot_period = plot_period

    def pltLoss(self, x, y, colors):
        if self.ax.lines:
            for line in self.ax.lines:
                line.set_xdata(x)
                line.set_ydata(y)
            self.ax.relim()      # make sure all the data fits
            self.ax.autoscale()  # auto-scale
        else:
            for color in colors:
                self.ax.plot(x, y, color)
                self.fig.show()
        self.fig.canvas.draw()


    def plot(self, i, losses):
        if i%self.plot_period==0:
            x = np.arange(len(losses))
            y = np.array(losses)
            self.pltLoss(x, y, ['C0'])
            plt.pause(0.0001)

