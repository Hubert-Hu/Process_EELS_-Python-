import matplotlib.pylab as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from copy import deepcopy
from sklearn.decomposition import NMF, PCA

class Line:

    def __init__(self, data, name=''):
        """
        data: a list, data[0] contains x values, data[1] contains y values.
        name: a string, used to label the name in the plotting.
        height: a number, the max height of zlp.
        """
        self.data = data
        self.name = name
        self.height = 0

    def slice_data(self, xrange):
        """
        Slice the data into desired range in x-axis.
        """
        d1 = [x for x in self.data[0] if x >= min(xrange) and x <= max(xrange)]
        d2 = [self.data[1][i] for i, x in enumerate(self.data[0]) if x >= min(xrange) and x <= max(xrange)]
        return [np.array(d1), np.array(d2)]

    def plot(self):
        plt.plot(self.data[0], self.data[1])

    def yshift_data(self, yshift):
        """
        shift data in y-axis for a better display in the plotting.
        """
        self.data[1] = self.data[1] + yshift

    def spline(self, data):
        """
        # data: a list, data[0] contains x values, data[1] contains y values.
        get the spline function from the data
        used to sub-pixel accuracy processing, such as find_zlp_max, alignment, find_peak.
        """
        # setting of the spline function
        s0 = 0
        k0 = 3
        y_func = UnivariateSpline(data[0], data[1], s=s0, k=k0)
        return y_func

    def get_linspace(self, xrange, num=5000, step=0):
        """
        Get linspace data by setting number of points or step, for constructing x values.
        If both are set, consider step first.
        Default num = 5000.
        """
        if step > 0:
            num_points = (max(xrange) - min(xrange)) / step + 1
            return np.linspace(min(xrange), max(xrange), num_points)
        if num > 0:
            return np.linspace(min(xrange), max(xrange), num)

    def find_zlp_max(self, zlp_xrange):
        """
        find the energy zero point by locating the maximum of zlp
        zlp_xrange: the range in x-axis to find the peak of zlp.
        return: zlp position and height
        """
        y_spl = self.spline(self.data)
        x_zlp = self.get_linspace(zlp_xrange)  # or we can set step
        shift, height = x_zlp[np.argmax(y_spl(x_zlp))], max(y_spl(x_zlp))
        self.height = height  # update height after aligning
        return shift, height

    def align(self, zlp_xrange):
        """
        Align the line by setting center of ZLP to zero.
        Use spline function to avoid subpixel misalignment.
        zlp_xrange: the range in x-axis to find the peak of zlp.
        """
        shift, height = self.find_zlp_max(zlp_xrange)
        # print('ZLP position: {}'.format(shift))
        y_spl = self.spline(self.data)
        self.data = [self.data[0], y_spl(self.data[0] + shift)]  # Positive zlp shift means a red shift

        # Slice the data since the data between [max(self.data[0])-shift, max(self.data[0])] in x-axis are invalid.
        ### Pay attention here, aligning the line data will change the length of x or y data.
        self.data = self.slice_data([min(self.data[0]), max(self.data[0]) - shift])
        return self.data

    def integrate(self, xrange):
        """
        Numerical integration between the xrange area.
        xrange: [lower_limit, higher_limit]
        """
        y_spl = self.spline(self.data)
        result = quad(y_spl, min(xrange), max(xrange))
        return result[0]

    def find_peak(self, xrange, display_peaks=False):
        """
        vis_xrange: an array of two numbers, showing the lower and upper limit of region of interets.
        method: a string, showing the method to find peaks.
        """
        x, sm_y = self.denoise_LLR(0.02)
        y_spl = self.spline([x, sm_y])
        x_line = self.get_linspace(xrange)  # or we can set step
        peaks, dic = find_peaks(y_spl(x_line), height=0, distance=100)
        if len(peaks):
            peaks, dic = find_peaks(y_spl(x_line), height=0, distance=100, prominence=max(dic['peak_heights']) / 100)
        if display_peaks:
            plt.figure()
            plt.plot(x_line, y_spl(x_line))
            plt.plot(x_line[peaks], y_spl(x_line)[peaks], 'xr')
        return x_line[peaks], dic['peak_heights']

    def denoise_LLR(self, dE=0.02, ncomp=1):
        """
        Denoise data by PCA reconstruction
        dE: a number, showing the energy window of one block.
        ncomp: a number, showing the number of the components.
        """
        # block size need to be an odd number.
        myscale = abs(self.data[0][1] - self.data[0][0])
        nx = int(np.floor(dE / myscale))
        if nx % 2 == 0:
            nx = nx + 1
        print('Smooth by PCA.')
        print('Setting dE block: {} ev. Real dE: {} eV'.format(dE, nx * myscale))

        # Creating the blocks
        num_block = len(self.data[1]) - nx + 1
        myblock = np.zeros([num_block, nx])
        for i in range(num_block):
            myblock[i, :] = self.data[1][i: i + nx]

        ## Perform PCA & reconstruct the spectra##
        # get the non-negative portion of the dataset
        data_mat = np.abs(myblock)
        # model = NMF(n_components=ncomp, init='random', random_state=0)
        model = PCA(n_components=ncomp, random_state=0)
        W = model.fit_transform(data_mat)
        H = model.components_
        # Reconstruct spectra
        mytp = np.dot(np.transpose([W[:, 0]]), np.array([H[0, :]]))

        # Unfold spectra
        mydata = np.zeros(num_block)
        y_mean = np.mean(self.data[1][:num_block])
        for i in range(num_block):
            mydata[i] = mytp[i, nx // 2] + y_mean

        return [self.data[0][nx // 2: num_block + nx // 2], mydata]


class Lines:

    def __init__(self, elements=[]):
        """
        elements: a list of Line objects.
        """
        self.elements = elements
        self.heights = []
        self.PCA_coefficients = []
        self.PCA_components = []

    def set_initial_elements(self, ele):
        """
        We can update initial elements by set_initial_elements.
        """
        self.elements = ele

    def add_lines(self, new_lines):
        for new_line in new_lines:
            if new_line not in self.elements:
                self.elements.append(new_line)

    def del_lines(self, new_lines):
        for new_line in new_lines:
            while True:
                if new_line in self.elements:
                    self.elements.remove(new_line)
                else:
                    break

    def slice_data(self, xrange):
        """
        Slice the data for each element.
        """
        for i, e in enumerate(self.elements):
            self.elements[i].data = e.slice_data(xrange)

    def align(self, zlp_xrange):
        """
        Align each element in elements.
        Since aligning will resize the data from each element, we slice all the aligned data into same size.
        This function assumes that the original x-values for all the lines are the same.
        zlp_xrange: the range in xaxis to find the peak of zlp.
        """
        xmin_list = []
        xmax_list = []
        for i, e in enumerate(self.elements):
            self.elements[i].data = e.align(zlp_xrange)
            xmin_list.append(min(self.elements[i].data[0]))
            xmax_list.append(max(self.elements[i].data[0]))
            self.heights.append(e.height)  # save height of each element in self.heights for further normalization.
        xrange = [max(xmin_list), min(xmax_list)]
        for i, e in enumerate(self.elements):
            self.elements[i].data = e.slice_data([max(xmin_list), xrange])

    def normalize(self):
        """
        Normalize the lines by setting the height of heighest zlp as 1.
        Align data before normalization to get the height data.
        """
        if self.heights:
            max_height = max(self.heights)
            for i, e in enumerate(self.elements):
                self.elements[i].data[1] = e.data[1] / max_height
        else:
            print('Error: there is no height list. Please align data first to get the list of heights.')

    def PCA(self, num_comps=6):
        """
        Use Principle Component Analysis (PCA) / Non-nagetive Matrix Factorization (NMF) to decomposite the spectra.
        In the result,
        W contains coefficients for each components: each column is an intensity map for corresponding component.
        H contains the components.
        Remember: do align first before PCA.
        """
        # transfer the data in elements to a (m * n) matrix,
        # m is the number of lines, n is the channel number for each line
        data_2d = []
        for e in self.elements:
            data_2d.append(e.data[1])
        data_2d = np.array(data_2d)

        # get the non-negative portion of the dataset
        data_mat = np.abs(data_2d)
        model = NMF(n_components=num_comps, init='random', random_state=0)
        # model = PCA(n_components=num_comps, random_state=0)
        W = model.fit_transform(data_mat)
        H = model.components_

        # save the data into two lists, each element is for one component.
        coeff_list = []
        comp_list = []
        for i in range(num_comps):
            coeff_list.append(W[:, i])
            comp_list.append([self.elements[0].data[0], H[i, :]])
        self.PCA_coefficients = deepcopy(coeff_list)
        self.PCA_components = deepcopy(comp_list)
        self.PCA_ncomps = num_comps
        self.PCA_model = model

    def slice_display(self, xrange):
        """
        Display the integrals under each line in the slice range.
        xrange: [lower_limit, higher_limit], showing the slice range in x-axis.
        """
        int_list = []
        for e in self.elements:
            int_list.append(e.integrate(xrange))
        return int_list

    def make_plot(self, config_dic):
        """
        Details in config_dic:
        xrange: the range in x-axis to display
        yshift_list: a list, each element is a number (e.g. 50%)
                    showing the shift on y-axis
                    as a percentage relative to the ymax of first line.
        label_list: a list, each element is a string as the label for each line, '_nolegend_' for hiding the label.
        color_list: a list, each element is an array as the color for each line.

        find the default color by:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        All the lists should be in the same size.
        """
        config = {'xrange': [0, 2],
                  'yshift_list': [0] * len(self.elements),
                  'label_list': ['_nolegend_'] * len(self.elements),
                  'color_list': ['b'] * len(self.elements),
                  'label_fontsize': 20,
                  'tick_fontsize': 20,
                  'legend_fontsize': 20,
                  'line_width': 3,
                  }
        config.update(config_dic)

        # plot in a selected range of x-axis.
        plt.figure(figsize=config['figure_size'])
        ymax_list = []
        ymin_list = []
        for num, line in enumerate(self.elements):
            # shift data in y-xaxis & slice data in a selected range for a better display
            shift = config['yshift_list'][num] * max(self.elements[0].data[1])
            new_data = line.slice_data(config['xrange'])
            plt.plot(new_data[0], new_data[1] + shift,
                     label=config['label_list'][num],
                     color=config['color_list'][num],
                     linewidth=config['line_width'])
            # save ymax and ymin of each line for further adjustment of y-limit.
            ymax_list.append(max(new_data[1]) + shift)
            ymin_list.append(min(new_data[1]) + shift)

        # set parameters for the figure.
        plt.xlim(self.config['xrange'])
        ymax = max(ymax_list)
        ymin = min(ymin_list)
        plt.ylim([ymin * 1.1, ymax * 1.1])
        plt.xlabel('Energy Loss (eV)', fontsize=config['label_fontsize'])
        plt.ylabel('Counts', fontsize=config['label_fontsize'])
        plt.yticks([])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis and y-axis
            which='both',  # both major and minor ticks are affected
            labelsize=self.config['tick_fontsize'])
        plt.legend(fontsize=config['legend_fontsize'], loc=1)

    def find_peak(self, xrange):
        """
        Find peaks for each line.
        """
        self.peak_positions = []
        self.peak_heights = []
        for e in self.elements:
            peak, height = e.find_peak(xrange, display_peak=False)
            print('Peak positions for {}: {}'.format(e.name, peak))
            print('Height {}: {}'.format(e.name, height))
            self.peak_positions.append(peak)
            self.peak_heights.append(height)

    def subtract(self, sub, display_range, display_sub=False):
        """
        Subtract the substrate signal from each line by aligning and normalizing first.
        sub: a list of Line objects, containing the information of substrate line.
        The number of Line objects could be 1 or number of elements.
        display_range: [lower_limit, upper_limit], a range in x-axis to construct new data.
        """
        if len(sub) == 1:
            subs = sub * len(self.elements)
        if len(sub) != len(self.elements):
            print('The number of substrate line is not either 1, or same as the number of lines.')
            return []

        zlp_xrange = [min(sub[0].data[0]), max(sub[0].data[0])]
        new_elements = []
        for i, e in enumerate(self.elements):
            # get spline function for sub-pixel alignment & find ZLP position of substrate line
            y_sub = sub[i].spline(sub[i].data)
            shift0, height0 = sub[i].find_zlp_max(zlp_xrange)
            # get spline function for sub-pixel alignment & find ZLP position of each line
            y_data = e.spline(e.data)
            shift1, height1 = e.find_zlp_max(zlp_xrange)
            x_data = e.get_linspace(display_range)
            # subtract the substrate signal from each line. Normalize each line by setting ZLP height as 1.
            e.data = [x_data, y_data(x_data + shift1) / height1 - y_sub(x_data + shift0) / height0]
            new_elements.append(e)
            if display_sub:
                plt.figure()
                plt.plot(x_data, y_data(x_data + shift1) / height1, label='Raw Data')
                plt.plot(x_data, y_sub(x_data + shift0) / height0, label='sub Data')
                plt.legend()
        return new_elements


class Mapping(Lines):

    def set_initial_by_data(self, xdata, ydata):
        """
        Set data here. Eventually it will be transfer to a list of Line objects.
        xdata: an array (1 * k) for x values.
        ydata: an array (m * n * k),
        (m, n) are the numbers of pixels in two axes, k is the channel number for one line.

        """
        if len(ydata.shape) == 3:
            self.xdata = xdata
            self.ydata = ydata
            self.pixel_num_x = ydata.shape[0]
            self.pixel_num_y = ydata.shape[1]
            self.pixel_num_z = ydata.shape[2]
            ydata_2d = np.reshape(ydata, [ydata.shape[0] * ydata.shape[1], -1])
            data_list = []
            for i in range(ydata_2d.shape[0]):
                data_list.append(Line([xdata, ydata_2d[i, :]]))
            self.elements = deepcopy(data_list)
        else:
            print('Error: the size of ydata did not match. Please asign a three-dimensional data to ydata.')

    def plot(self):
        plt.imshow(self.ydata)

    def select_sum_all(self):
        newsp_x = self.elements[0].data[0]
        newsp_y = 0 * self.elements[0].data[1]
        for e in self.elements:
            newsp_y = newsp_y + e.data[1]
        return [newsp_x, newsp_y]

    def coord_to_row(self, select):
        return select[0] * self.pixel_num_y + select[1]

    def select_sum_by_list(self, select_list):
        index = self.coord_to_row(select_list[0])
        newsp_x = self.elements[index].data[0]
        newsp_y = 0 * self.elements[index].data[1]
        for s in select_list:
            index = self.coord_to_row(s)
            newsp_y = newsp_y + self.elements[index].data[1]
        newsp_y = newsp_y / len(select_list)
        return [newsp_x, newsp_y]

    def initial_process(self):
        """
        Process data in the order of alignment and normalization.
        """
        xdata = self.elements[0].data[0]
        zlp_xrange = [min(xdata), max(xdata)]
        self.align(zlp_xrange)
        self.normalize()
        self.xdata = self.elements[0].data[0]

    def PCA_plot(self, file_prefix=''):
        """
        Plot each components and corresponding coefficient mapping.
        """
        import matplotlib.gridspec as gridspec
        for i in range(self.PCA_ncomps):
            self.PCA_coefficients[i] = np.reshape(self.PCA_coefficients[i], [self.pixel_num_x, self.pixel_num_y])
            plt.figure(figsize=[8, 3])
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 6])
            plt.subplot(gs[0])
            plt.imshow(self.PCA_coefficients[i])
            plt.subplot(gs[1])
            plt.plot(self.PCA_components[i][0], self.PCA_components[i][1])
            plt.xlabel('Energy Loss (eV)')
            plt.tight_layout()
            plt.savefig(file_prefix + 'PCA_#{}.png'.format(i), bbox_inches='tight')
