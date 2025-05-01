import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class UMAPView(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots()
        super(UMAPView, self).__init__(fig)
        self.setParent(parent)
        self.ax.set_title('UMAP of Feature Vectors')
        self.on_point_click = None
        self.cid = self.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.inaxes is not None and self.on_point_click:
            x, y = event.xdata, event.ydata
            self.on_point_click(x, y)

    def plot(self, embedding, labels, highlight_index):
        if embedding is None:
            return None, None
        
        self.ax.clear()
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        
        for label, color in zip(unique_labels, colors):
            self.ax.scatter(embedding[labels == label, 0], embedding[labels == label, 1], 
                          c=[color], label=f'Cluster {label}')
        
        if 0 <= highlight_index < len(embedding):
            current_label = labels[highlight_index]
            highlight_color = label_to_color[current_label]
            
            self.ax.scatter(embedding[highlight_index, 0], embedding[highlight_index, 1], 
                          c=[highlight_color], edgecolors='black', s=100, label='Current instance')
            
            self.ax.legend()
        
        self.draw()
        
        return label_to_color, None if highlight_index >= len(labels) else labels[highlight_index]

    def set_point_click_handler(self, handler):
        self.on_point_click = handler