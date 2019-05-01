# Process_EELS_-Python-
Process EELS systematically: align, normalize, PCA, denoise_LLR, plot, find peak.

The main script is 'Process_EELS.py', which includes three classes:  

1. **Line**.  
The instance of the class is an object describing a line of EELS.  
Main function: **slice, plot, find_zlp, align, raise_in_y, spline, integrate, find_peak, denoise_LLR**.

2. **Lines**.  
The instance of the class is an object describing a list of EELS lines.  
Main function: **add, delete, slice, slice_display, align, normalize, PCA, make_plot, find_peak, subtract background signals**.

3. **Map**.  
The Child Class of **Lines**.  
Additional function: **sum_in_an_area, mapping_slice_display, PCA_plot**

More examples using these two classes for systematically analyze EELS data are presented.  
