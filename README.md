# Process_EELS_-Python-
Process EELS systematically: align, normalize, PCA, denoise_LLR, plot, find peak.

The main script is 'Process_EELS.py', which includes three classes:  

1. **Line**.  
The instance of the class is an object describing a line of EELS.  
Main function: **slice, plot, find_zlp, align, yshift, spline, integrate, find_peak, denoise_LLR**.

2. **Lines**.  
The instance of the class is an object describing a list of EELS lines.  
Main function: **add, delete, slice, slice_display, align, normalize, denoise_LLR, PCA, make_plot, find_peak, subtract background signals**.

3. **Map**.  
The Child Class of **Lines**.  
Additional function: **normalize_map,sum_in_an_area, mapping_slice_display, PCA_plot**

Examples using these three classes for systematically analyze EELS data are presented:

1. **'Process_EELS_Line.ipynb'**  
Examples of the basic functions based on Line Class:  
1: slice_data  
2: yshift_data  
3: spline  
4: find_zlp_max  
5: denoise_LLR  
6: find_peak

2. **'Process_EELS_Lines.ipynb'**  
Examples of the further analysis based on Lines Class:  
1: Plot multiple lines.  
2: Validation test of the manual background subtraction by finding peaks.  
3: Background subtraction by raw and subtrate data. 

3. **'Process_EELS_Map.ipynb'**  
Examples of the EELS mapping analysis based on Map Class:  
1: Initial process (align, and normalize).  
2: Slice display.  
3: Decomposition by PCA (or NMF).  
4: Summming over all the pixels.  

Any feedback is welcome. Please let me know if you have any questions on bugs or any suggestions to expand the functionality.

If you like it, don’t forget to give it a star.
