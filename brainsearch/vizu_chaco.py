from traits.api import HasTraits, Instance, Property, Event, Tuple, cached_property
from traits.api import Float, Enum, List, Range, Int, Array, Bool
from traitsui.api import Item, UItem, View, Group, Handler, CheckListEditor, RangeEditor
from enable.api import Component, ComponentEditor, KeySpec, BaseTool

from chaco.api import (ArrayPlotData, Plot, gray, HPlotContainer, BarPlot, DataView,
                       ColorBar, LinearMapper, GridPlotContainer, VPlotContainer, LabelAxis,
                       Legend, DataRange1D, jet, hot, ImagePlot)
from chaco.tools.api import PanTool, RangeSelection, RangeSelectionOverlay, ZoomTool, RangeSelection2D
from chaco.tools.image_inspector_tool import ImageInspectorTool, ImageInspectorOverlay
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool

from time import time
import numpy as np

from enable.api import ColorTrait
from traits.api import Float
from chaco.api import AbstractOverlay, BaseXYPlot


class ZoomOverlay(AbstractOverlay):
    """
    Draws a trapezoidal selection overlay from the source plot to the
    destination plot.  Assumes that the source plot lies above the destination
    plot.
    """

    source = Instance(Component)
    destination = Instance(Component)

    border_color = ColorTrait((0, 0, 0.7, 1))
    border_width = Int(1)
    fill_color = ColorTrait("dodgerblue")
    alpha = Float(0.3)

    def calculate_points(self, component):
        """
        Calculate the overlay polygon based on the selection and the location
        of the source and destination plots.
        """
        # find selection range on source plot
        x_start, x_end = self._get_selection_screencoords()
        if x_start > x_end:
            x_start, x_end = x_end, x_start

        y_end = self.source.y
        y_start = self.source.y2

        left_top = np.array([x_start, y_start])
        left_mid = np.array([x_start, y_end])
        right_top = np.array([x_end, y_start])
        right_mid = np.array([x_end, y_end])

        # Offset y because we want to avoid overlapping the trapezoid with the topmost
        # pixels of the destination plot.
        y = self.destination.y2 + 1

        left_end = np.array([self.destination.x, y])
        right_end = np.array([self.destination.x2, y])

        polygon = np.array((left_top, left_mid, left_end,
                         right_end,right_mid, right_top))
        left_line = np.array((left_top, left_mid, left_end))
        right_line = np.array((right_end,right_mid, right_top))

        return left_line, right_line, polygon

    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        """
        Draws this overlay onto 'component', rendering onto 'gc'.
        """

        tmp = self._get_selection_screencoords()
        if tmp is None:
            return

        left_line, right_line, polygon = self.calculate_points(component)

        with gc:
            gc.translate_ctm(*component.position)
            gc.set_alpha(self.alpha)
            gc.set_fill_color(self.fill_color_)
            gc.set_line_width(self.border_width)
            gc.set_stroke_color(self.border_color_)
            gc.begin_path()
            gc.lines(polygon)
            gc.fill_path()

            gc.begin_path()
            gc.lines(left_line)
            gc.lines(right_line)
            gc.stroke_path()

        return

    def _get_selection_screencoords(self):
        """
        Returns a tuple of (x1, x2) screen space coordinates of the start
        and end selection points.  If there is no current selection, then
        returns None.
        """
        selection = self.source.index.metadata["selections"]
        if selection is not None and len(selection) == 2:
            mapper = self.source.index_mapper
            return mapper.map_screen(np.array(selection))
        else:
            return None

    #------------------------------------------------------------------------
    # Trait event handlers
    #------------------------------------------------------------------------

    def _source_changed(self, old, new):
        if old is not None and old.controller is not None:
            old.controller.on_trait_change(self._selection_update_handler, "selection",
                                           remove=True)
        if new is not None and new.controller is not None:
            new.controller.on_trait_change(self._selection_update_handler, "selection")
        return

    def _selection_update_handler(self, value):
        if value is not None and self.destination is not None:
            r = self.destination.index_mapper.range
            start, end = np.amin(value), np.amax(value)
            r.low = start
            r.high = end

        self.source.request_redraw()
        self.destination.request_redraw()
        return


class CursorTool2D(BaseCursorTool):
    current_index = Tuple(0,0)

    def _current_index_changed(self):
        self.component.request_redraw()

    @cached_property
    def _get_current_position(self):
        plot = self.component
        ndx, ndy = self.current_index
        xdata, ydata = plot.index.get_data()
        x = xdata.get_data()[ndx]
        y = ydata.get_data()[ndy]
        return x,y

    def _set_current_position(self, traitname, args):
        plot = self.component
        xds, yds = plot.index.get_data()
        ndx = xds.reverse_map(args[0])
        ndy = yds.reverse_map(args[1])
        if ndx is not None and ndy is not None:
            self.current_index = ndx, ndy

    def draw(self, gc, view_bounds=None):
        """ Draws this tool on a graphics context.

        Overrides LineInspector, BaseTool.
        """
        # We draw at different points depending on whether or not we are
        # interactive.  If both listener and interactive are true, then the
        # selection metadata on the plot component takes precendence.
        plot = self.component
        if plot is None:
            return
        sx, sy = plot.map_screen([np.array(self.current_position)+0.5])[0]
        orientation = plot.orientation

        if orientation == "h":
            if sx is not None:
                self._draw_vertical_line(gc, sx)
            if sy is not None:
                self._draw_horizontal_line(gc, sy)
        else:
            if sx is not None:
                self._draw_horizontal_line(gc, sx)
            if sy is not None:
                self._draw_vertical_line(gc, sy)

        if self.show_marker and sx is not None and sy is not None:
            self._draw_marker(gc, sx, sy)

    def normal_left_down(self, event):
        x,y = event.x, event.y
        plot = self.component
        ndx = plot.map_index((x, y), threshold=0.0, index_only=True)
        if ndx is None:
            return
        newx, newy = self.current_index
        if ndx[0] is not None:
            newx = ndx[0]
        if ndx[1] is not None:
            newy = ndx[1]
        self.current_index = newx, newy
        #print self.current_index
        plot.request_redraw()


class ImageSelectorTool(BaseTool):
    """ A tool that captures the color and underlying values of an image plot.
    """

    # This event fires whenever the mouse clicks on a new image point.
    # Its value is a dict with a key "color_value", and possibly a key
    # "data_value" if the plot is a color-mapped image plot.
    new_value = Event

    # Indicates whether overlays listening to this tool should be visible.
    visible = Bool(True)

    # Stores the last mouse position.  This can be used by overlays to
    # position themselves around the mouse.
    last_mouse_position = Tuple

    # This key will show and hide any ImageInspectorOverlays associated
    # with this tool.
    inspector_key = KeySpec('p')

    # Stores the value of self.visible when the mouse leaves the tool,
    # so that it can be restored when the mouse enters again.
    _old_visible = Enum(None, True, False) #Trait(None, Bool(True))

    def normal_key_pressed(self, event):
        if self.inspector_key.match(event):
            self.visible = not self.visible
            event.handled = True

    def normal_mouse_leave(self, event):
        if self._old_visible is None:
            self._old_visible = self.visible
            self.visible = False

    def normal_mouse_enter(self, event):
        if self._old_visible is not None:
            self.visible = self._old_visible
            self._old_visible = None

    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed.

        Fires the **new_value** event with the data (if any) from the event's
        position.
        """
        plot = self.component
        if plot is not None:
            if isinstance(plot, ImagePlot):
                ndx = plot.map_index((event.x, event.y))
                if ndx == (None, None):
                    self.new_value = None
                    return

                x_index, y_index = ndx
                image_data = plot.value
                if hasattr(plot, "_cached_mapped_image") and \
                       plot._cached_mapped_image is not None:
                    self.new_value = \
                            dict(indices=ndx,
                                 data_value=image_data.data[y_index, x_index],
                                 color_value=plot._cached_mapped_image[y_index,
                                                                       x_index])

                else:
                    self.new_value = \
                        dict(indices=ndx,
                             color_value=image_data.data[y_index, x_index])

                self.last_mouse_position = (event.x, event.y)
                print self.last_mouse_position
        return


class BrainsearchViewer(HasTraits):
    k = Range(low=1, high=100, value=1)

    #image_selector_tool = Instance(ImageSelectorTool)
    cursor = Instance(CursorTool2D)
    cursor2 = Instance(CursorTool2D)

    hist = Instance(Plot)
    stackedhist = Instance(Component)
    hist2d = Instance(Component)

    brain_voxels = Array
    brain = Instance(Component)
    axial = Range(value=50, low=0, high_name='_axial_max')

    _axial_max = Property(Int, depends_on=['brain_voxels'])

    traits_view = View(Group(Group(Item('k'),
                                   Group(Item('hist', editor=ComponentEditor(size=(70, 20)), show_label=False),
                                         Item('hist2d', editor=ComponentEditor(size=(70, 20)), show_label=False),
                                         orientation="horizontal"),
                                   orientation="vertical"),
                             orientation="horizontal"),
                       Item('stackedhist', editor=ComponentEditor(size=(136, 20)), show_label=False),
                       resizable=True,
                       width=1366//2, height=768//2,
                       title="Brainsearch Viewer",
                       )

    brain_view = View(Item('axial', editor=RangeEditor(low=0, high_name="_axial_max", mode="slider")),
                      Item('brain', editor=ComponentEditor(size=(100, 100)), show_label=False),
                      resizable=True,
                      width=400, height=400,
                      title="Brain Viewer",
                      )

    def _get__axial_max(self):
        return self.brain_voxels.shape[2]-1

    def __init__(self, query, neighbors, **kwargs):
        super(BrainsearchViewer, self).__init__(**kwargs)

        self.query = query
        self.half_vovel_size = np.array(self.query['patch_size'])//2

        starttime = time()
        self.nb_neighbors = query['nb_neighbors']

        # Hist
        self.labels = neighbors['labels']
        self.positives = np.cumsum(self.labels == 1, axis=1)
        self.negatives = np.cumsum(self.labels == 0, axis=1)

        positives = self.positives[:, self.k-1]
        negatives = self.negatives[:, self.k-1]
        minlength = max(positives.max(), negatives.max()) + 1
        positives = np.bincount(positives, minlength=minlength)
        negatives = np.bincount(negatives, minlength=minlength)

        self.hist_data = ArrayPlotData(index=range(minlength))
        self.hist_data.set_data("positives", positives)
        self.hist_data.set_data("negatives", negatives)

        # Stacked Hist
        self.patches_per_axial_slice = {}
        for i in range(self.brain_voxels.shape[2]):
            self.patches_per_axial_slice[i] = np.where(query['positions'][:, 2] == i)[0]

        negatives = self.negatives[:, self.k-1]
        positives = negatives + self.positives[:, self.k-1]
        selected_axial = self.axial - self.half_vovel_size[2]
        self.stackedhist_data = ArrayPlotData(index=range(len(self.patches_per_axial_slice[selected_axial])))
        self.stackedhist_data.set_data("selected", np.zeros(len(self.patches_per_axial_slice[selected_axial]), dtype="float32"))
        self.stackedhist_data.set_data("positives", positives[self.patches_per_axial_slice[selected_axial]])
        self.stackedhist_data.set_data("negatives", negatives[self.patches_per_axial_slice[selected_axial]])

        # Hist 2D
        self.pos_dist = np.ones((len(self.nb_neighbors), self.k), dtype=np.int32) * np.nan
        self.metric_dist = np.ones((len(self.nb_neighbors), self.k), dtype=np.int32) * np.nan

        self.pos_dist = np.sqrt(np.nansum((neighbors['positions'] - query['positions'][:, None, :])**2, axis=2))
        self.metric_dist = neighbors['distances']

        nbins = 100
        self.H = np.zeros((100, nbins, nbins))

        self.xedges = np.linspace(0, np.nanmax(self.pos_dist), num=nbins+1)
        self.yedges = np.linspace(0, np.nanmax(self.metric_dist), num=nbins+1)
        for i in range(100):
            pos_dist = self.pos_dist[:, i]
            metric_dist = self.metric_dist[:, i]
            idx = np.isfinite(pos_dist)
            H, xedges, yedges = np.histogram2d(pos_dist[idx], metric_dist[idx], bins=[self.xedges, self.yedges])
            self.H[i] = H.T  # Transpose since histogram2d consider the x as the first dimension and y as the second dimension.

        self.H = np.cumsum(self.H, axis=0)
        self.H = np.log(self.H)
        self.H[np.isinf(self.H)] = np.nan

        self.hist2d_data = ArrayPlotData()
        self.hist2d_data.set_data("H", self.H[self.k-1])

        # Brain
        self.brain_data = ArrayPlotData()
        self.brain_data.set_data("axial", self.brain_voxels[:, :, self.axial].T)

        # Brain map
        OFFSET = 0.01
        self.classif_map = np.zeros_like(self.brain_voxels, dtype=np.float32)
        pixel_pos = query['positions'] + self.half_vovel_size

        nb_neighbors_per_patch = np.minimum(self.nb_neighbors, self.k)
        nb_pos = np.nansum(self.labels[:, :self.k], dtype=np.float64, axis=1)
        self.classif_map[zip(*pixel_pos)] = nb_pos/nb_neighbors_per_patch + OFFSET
        self.classif_map[np.isnan(self.classif_map)] = 0.5 + OFFSET

        self.classif_map[self.classif_map == 0.] = np.nan
        self.brain_data.set_data("map_axial", self.classif_map[:, :, self.axial].T)
        print "Init:", time() - starttime

    def _hist_default(self):
        plot = Plot(self.hist_data, padding=(50, 0, 0, 40))
        plot_control = plot.plot(('index', 'negatives'), type='bar', bar_width=0.8, fill_color='blue', alpha=0.5)
        plot_parkinson = plot.plot(('index', 'positives'), type='bar', bar_width=0.8, fill_color='yellow', alpha=0.5)

        # set the plot's value low range to 0, otherwise it will pad too much
        plot.value_range = DataRange1D(low_settings=0)

        plot.index_axis.title = "Class count / patch"
        legend = Legend(componeent=plot, padding=5, align='ur',
                        plots={'Control': plot_control, 'Parkinson': plot_parkinson})
        plot.overlays.append(legend)
        return plot

    def _stackedhist_default(self):
        plot = Plot(self.stackedhist_data, padding=(25, 0, 5, 20))
        plot_selected = plot.plot(('index', 'selected'), type='bar', bar_width=1, fill_color='red', line_color='red')
        plot_parkinson = plot.plot(('index', 'positives'), type='bar', bar_width=0.5, fill_color='yellow', line_color='yellow')
        plot_control = plot.plot(('index', 'negatives'), name="select", type='bar', bar_width=0.5, fill_color='blue', line_color='blue')

        # set the plot's value low range to 0, otherwise it will pad too much
        plot.value_range = DataRange1D(low_settings=0)

        legend = Legend(component=plot, padding=5, align='ur',
                        plots={'Control': plot_control, 'Parkinson': plot_parkinson})
        plot.overlays.append(legend)
        #return plot

        zoomplot = Plot(self.stackedhist_data, padding=(25, 0, 5, 20))
        zoomplot_selected = zoomplot.plot(('index', 'selected'), type='bar', bar_width=1, fill_color='red', line_color='red')
        zoomplot_parkinson = zoomplot.plot(('index', 'positives'), type='bar', bar_width=0.5, fill_color='yellow', line_color='yellow')
        zoomplot_control = zoomplot.plot(('index', 'negatives'), type='bar', bar_width=0.5, fill_color='blue', line_color='blue')

        zoomplot.value_range = DataRange1D(low_settings=0)
        zoomplot.index_axis.title = "Class count / patch"

        outer_container = VPlotContainer(padding=20,
                                         fill_padding=True,
                                         spacing=0,
                                         stack_order='top_to_bottom',
                                         bgcolor='lightgray',
                                         use_backbuffer=True)

        plot.index = plot_control[0].index

        outer_container.add(plot)
        outer_container.add(zoomplot)
        plot.controller = RangeSelection(plot)
        zoom_overlay = ZoomOverlay(source=plot, destination=zoomplot)
        outer_container.overlays.append(zoom_overlay)

        return outer_container

    def _hist2d_default(self):
        plot = Plot(self.hist2d_data, padding=(20, 0, 0, 40))
        plot.img_plot("H", xbounds=self.xedges, ybounds=self.yedges, colormap=jet)
        plot.index_axis.title = "Voxel dist."
        plot.value_axis.title = "Root Square Error"

        # Create a colorbar
        colormap = plot.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            padding=(20, 30, 0, 0))
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        # Create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer=True, padding=0)
        container.add(colorbar)
        container.add(plot)
        container.bgcolor = "lightgray"

        return container

    def _brain_default(self):
        plot = Plot(self.brain_data, padding=0)
        plot.width = self.brain_voxels.shape[1]
        plot.height = self.brain_voxels.shape[0]
        plot.aspect_ratio = 1.
        plot.index_axis.visible = False
        plot.value_axis.visible = False
        renderer = plot.img_plot("axial", colormap=gray)[0]
        plot.color_mapper.range = DataRange1D(low=0., high=1.0)
        plot.bgcolor = 'pink'

        # Brain tools
        plot.tools.append(PanTool(plot, drag_button="right"))
        plot.tools.append(ZoomTool(plot))
        imgtool = ImageInspectorTool(renderer)
        renderer.tools.append(imgtool)
        overlay = ImageInspectorOverlay(component=renderer, image_inspector=imgtool,
                                        bgcolor="white", border_visible=True)
        renderer.overlays.append(overlay)

        # Brain track cursor
        self.cursor = CursorTool2D(renderer, drag_button='left', color='red', line_width=2.0)
        self.cursor.on_trait_change(self.update_stackedhist, 'current_index')
        self.cursor.current_position = (0., 0.)
        renderer.overlays.append(self.cursor)

        # Brain colorbar
        colormap = plot.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            padding=(30, 0, 0, 0))
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        # Brain_map
        plot2 = Plot(self.brain_data, padding=0)
        plot2.width = self.brain_voxels.shape[1]
        plot2.height = self.brain_voxels.shape[0]
        plot2.aspect_ratio = 1.
        plot2.index_axis.visible = False
        plot2.value_axis.visible = False
        renderer2 = plot2.img_plot("map_axial", colormap=gray)[0]
        plot2.color_mapper.range = DataRange1D(low=0., high=1.0)
        plot2.bgcolor = 'pink'
        plot2.range2d = plot.range2d

        # Brain_map tools
        plot2.tools.append(PanTool(plot2, drag_button="right"))
        plot2.tools.append(ZoomTool(plot2))
        imgtool2 = ImageInspectorTool(renderer2)
        renderer2.tools.append(imgtool2)
        overlay2 = ImageInspectorOverlay(component=renderer2, image_inspector=imgtool2,
                                        bgcolor="white", border_visible=True)
        renderer2.overlays.append(overlay2)

        # Brain_map track cursor
        self.cursor2 = CursorTool2D(renderer2, drag_button='left', color='red', line_width=2.0)
        #self.cursor2.on_trait_change(self.update_stackedhist, 'current_index')
        self.cursor2.on_trait_change(self.cursor2_changed, 'current_index')
        self.cursor2.current_position = (0., 0.)
        renderer2.overlays.append(self.cursor2)

        # Brain_map colorbar
        colormap2 = plot2.color_mapper
        colorbar2 = ColorBar(index_mapper=LinearMapper(range=colormap2.range),
                             color_mapper=colormap2,
                             plot=plot2,
                             orientation='v',
                             resizable='v',
                             width=20,
                             padding=(30, 0, 0, 0))
        colorbar2.padding_top = plot2.padding_top
        colorbar2.padding_bottom = plot2.padding_bottom

        # Create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer=True, padding=(0, 0, 10, 10))
        container.add(plot)
        container.add(colorbar)
        container.bgcolor = "lightgray"

        container2 = HPlotContainer(use_backbuffer=True, padding=(0, 0, 10, 10))
        container2.add(plot2)
        container2.add(colorbar2)
        container2.bgcolor = "lightgray"

        Vcontainer = VPlotContainer(use_backbuffer=True)
        Vcontainer.add(container2)
        Vcontainer.add(container)
        Vcontainer.bgcolor = "lightgray"

        return Vcontainer

    def _k_changed(self):
        positives = self.positives[:, self.k-1]
        negatives = self.negatives[:, self.k-1]
        minlength = max(positives.max(), negatives.max()) + 1
        positives = np.bincount(positives, minlength=minlength)
        negatives = np.bincount(negatives, minlength=minlength)

        self.hist_data.set_data("index", range(minlength))
        self.hist_data.set_data("positives", positives)
        self.hist_data.set_data("negatives", negatives)

        # Stacked Hist
        negatives = self.negatives[:, self.k-1]
        positives = negatives + self.positives[:, self.k-1]
        selected_axial = self.axial - self.half_vovel_size[2]
        self.stackedhist_data.set_data("positives", positives[self.patches_per_axial_slice[selected_axial]])
        self.stackedhist_data.set_data("negatives", negatives[self.patches_per_axial_slice[selected_axial]])

        #Hist 2D
        self.hist2d_data.set_data("H", self.H[self.k-1])

        # Brain_map
        OFFSET = 0.01
        self.classif_map = np.zeros_like(self.brain_voxels, dtype=np.float32)
        pixel_pos = self.query['positions'] + self.half_vovel_size

        nb_neighbors_per_patch = np.minimum(self.nb_neighbors, self.k)
        nb_pos = np.nansum(self.labels[:, :self.k], dtype=np.float64, axis=1)
        self.classif_map[zip(*pixel_pos)] = nb_pos/nb_neighbors_per_patch + OFFSET
        self.classif_map[np.isnan(self.classif_map)] = 0.5 + OFFSET
        self.classif_map[self.classif_map == 0.] = np.nan
        self.brain_data.set_data("map_axial", self.classif_map[:, :, self.axial].T)

    def _axial_changed(self):
        self.axial = min(self.axial, self.brain_voxels.shape[2]-1)
        self.axial = max(self.axial, 0)

        self.brain_data.set_data("axial", self.brain_voxels[:, :, self.axial].T)
        self.brain_data.set_data("map_axial", self.classif_map[:, :, self.axial].T)

        # Stacked Hist
        negatives = self.negatives[:, self.k-1]
        positives = negatives + self.positives[:, self.k-1]
        if self.axial < self.half_vovel_size[2] or self.axial > (self.brain_voxels.shape[2]+self.half_vovel_size[2]-1):
            self.stackedhist_data.set_data("index", [])
            self.stackedhist_data.set_data("positives", [])
            self.stackedhist_data.set_data("negatives", [])
            self.stackedhist_data.set_data("selected", [])
        else:
            selected_axial = self.axial - self.half_vovel_size[2]
            self.stackedhist_data.set_data("index", range(len(self.patches_per_axial_slice[selected_axial])))
            self.stackedhist_data.set_data("positives", positives[self.patches_per_axial_slice[selected_axial]])
            self.stackedhist_data.set_data("negatives", negatives[self.patches_per_axial_slice[selected_axial]])
            self.update_stackedhist()

    def cursor2_changed(self):
        self.cursor.current_index = self.cursor2.current_index
        self.cursor._current_index_changed()

    def update_stackedhist(self):
        print self.cursor.current_index
        self.cursor2.current_index = self.cursor.current_index
        self.cursor2._current_index_changed()

        # Patches are indexed by their corner (0,0,0) so we have to substract half of the patch's size
        #  to the selected voxel indice.
        selected_position = np.array(self.cursor.current_index + (self.axial,)) - self.half_vovel_size

        # Stacked Hist
        matches_axial = self.query['positions'][self.query['positions'][:, 2] == selected_position[2]]
        selection = np.all(matches_axial == selected_position, axis=1).astype('float32')
        print selection.sum(), "matches"
        self.stackedhist_data.set_data("selected", selection * self.k)

        # Zoom at the right place
        if len(np.where(selection)[0]) > 0:
            selected_index = np.where(selection)[0][0]
            new_selection = (max(0., selected_index-50.), min(float(len(selection)), selected_index+50.))
            self.stackedhist.components[0].controller._set_selection(new_selection)

    def configure_traits(self, *args, **kwargs):
        self.edit_traits("brain_view")
        super(BrainsearchViewer, self).configure_traits(*args, **kwargs)


class NoisyBrainsearchViewer(HasTraits):
    noise = Range(low=0., high=1., values=np.linspace(0, 1, 100))

    cursor = Instance(CursorTool2D)
    cursor2 = Instance(CursorTool2D)

    brain_voxels = Array
    brain = Instance(Component)
    axial = Range(value=50, low=0, high_name='_axial_max')
    _axial_max = Property(Int, depends_on=['brain_voxels'])

    traits_view = View(Item('axial', editor=RangeEditor(low=0, high_name="_axial_max", mode="slider")),
                       Item('brain', editor=ComponentEditor(size=(100, 100)), show_label=False),
                       Item('noise', label="Noise (std)"),
                       resizable=True,
                       width=400, height=400,
                       title="Noisy Brain Viewer",
                       )

    def _get__axial_max(self):
        return self.brain_voxels.shape[2]-1

    def __init__(self, query, engine, **kwargs):
        super(NoisyBrainsearchViewer, self).__init__(**kwargs)

        self.query = query
        self.engine = engine

        starttime = time()
        self.codes = self.engine.lshashes[0].hash_vector(self.query['patches'])

        # Brain
        self.brain_data = ArrayPlotData()
        self.brain_data.set_data("axial", self.brain_voxels[:, :, self.axial].T)

        self.brain_mask = (self.brain_voxels != 0).astype('float32')
        self.noise_image = self.noise * np.random.randn(*self.brain_voxels.shape) * self.brain_mask
        self.brain_data.set_data("noisy_axial", self.brain_voxels[:, :, self.axial].T + self.noise_image[:, :, self.axial].T)

        print "Init:", time() - starttime

    def _brain_default(self):
        plot = Plot(self.brain_data, padding=0)
        plot.width = self.brain_voxels.shape[1]
        plot.height = self.brain_voxels.shape[0]
        plot.aspect_ratio = 1.
        plot.index_axis.visible = False
        plot.value_axis.visible = False
        renderer = plot.img_plot("axial", colormap=gray)[0]
        plot.color_mapper.range = DataRange1D(low=0., high=1.0)
        plot.bgcolor = 'pink'

        # Brain tools
        plot.tools.append(PanTool(plot, drag_button="right"))
        plot.tools.append(ZoomTool(plot))
        imgtool = ImageInspectorTool(renderer)
        renderer.tools.append(imgtool)
        overlay = ImageInspectorOverlay(component=renderer, image_inspector=imgtool,
                                        bgcolor="white", border_visible=True)
        renderer.overlays.append(overlay)

        # Brain track cursor
        self.cursor = CursorTool2D(renderer, drag_button='left', color='red', line_width=2.0)
        #self.cursor.on_trait_change(self.update_stackedhist, 'current_index')
        self.cursor.current_positionyou = (0., 0.)
        renderer.overlays.append(self.cursor)

        # Brain colorbar
        colormap = plot.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            padding=(30, 0, 0, 0))
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        # Noisy brain
        plot2 = Plot(self.brain_data, padding=0)
        plot2.width = self.brain_voxels.shape[1]
        plot2.height = self.brain_voxels.shape[0]
        plot2.aspect_ratio = 1.
        plot2.index_axis.visible = False
        plot2.value_axis.visible = False
        renderer2 = plot2.img_plot("noisy_axial", colormap=gray)[0]
        plot2.color_mapper.range = DataRange1D(low=0., high=1.0)
        plot2.bgcolor = 'pink'
        plot2.range2d = plot.range2d

        # Brain_map tools
        plot2.tools.append(PanTool(plot2, drag_button="right"))
        plot2.tools.append(ZoomTool(plot2))
        imgtool2 = ImageInspectorTool(renderer2)
        renderer2.tools.append(imgtool2)
        overlay2 = ImageInspectorOverlay(component=renderer2, image_inspector=imgtool2,
                                         bgcolor="white", border_visible=True)
        renderer2.overlays.append(overlay2)

        # Brain_map track cursor
        self.cursor2 = CursorTool2D(renderer2, drag_button='left', color='red', line_width=2.0)
        #self.cursor2.on_trait_change(self.cursor2_changed, 'current_index')
        self.cursor2.current_position = (0., 0.)
        renderer2.overlays.append(self.cursor2)

        # Brain_map colorbar
        colormap2 = plot2.color_mapper
        colorbar2 = ColorBar(index_mapper=LinearMapper(range=colormap2.range),
                             color_mapper=colormap2,
                             plot=plot2,
                             orientation='v',
                             resizable='v',
                             width=20,
                             padding=(30, 0, 0, 0))
        colorbar2.padding_top = plot2.padding_top
        colorbar2.padding_bottom = plot2.padding_bottom

        # Create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer=True, padding=(0, 0, 10, 10))
        container.add(plot)
        container.add(colorbar)
        container.bgcolor = "lightgray"

        container2 = HPlotContainer(use_backbuffer=True, padding=(0, 0, 10, 10))
        container2.add(plot2)
        container2.add(colorbar2)
        container2.bgcolor = "lightgray"

        Hcontainer = HPlotContainer(use_backbuffer=True)
        Hcontainer.add(container)
        Hcontainer.add(container2)
        Hcontainer.bgcolor = "lightgray"

        return Hcontainer

    def _noise_changed(self):
        # Noisy brain
        self.noise_image = self.noise * np.random.randn(*self.brain_voxels.shape) * self.brain_mask
        self.brain_data.set_data("noisy_axial", self.brain_voxels[:, :, self.axial].T + self.noise_image[:, :, self.axial].T)

        # Compute codes and compare with original
        #starttime = time()
        noisy_patches = self.query['patches'].copy()
        for x in range(noisy_patches.shape[1]):
            for y in range(noisy_patches.shape[2]):
                for z in range(noisy_patches.shape[3]):
                    posX = self.query['positions'][:, 0] + x
                    posY = self.query['positions'][:, 1] + y
                    posZ = self.query['positions'][:, 2] + z
                    noisy_patches[:, x, y, z] += self.noise_image[posX, posY, posZ]

        #print time() - starttime
        noisy_codes = self.engine.lshashes[0].hash_vector(noisy_patches)
        print "{:.2f}%".format(np.mean(np.array(self.codes) == np.array(noisy_codes))*100)

    def _axial_changed(self):
        self.axial = min(self.axial, self.brain_voxels.shape[2]-1)
        self.axial = max(self.axial, 0)

        self.brain_data.set_data("axial", self.brain_voxels[:, :, self.axial].T)
        self.brain_data.set_data("noisy_axial", self.brain_voxels[:, :, self.axial].T + self.noise_image[:, :, self.axial].T)

    def cursor2_changed(self):
        self.cursor.current_index = self.cursor2.current_index
        self.cursor._current_index_changed()
