// Source file for the depth image solver program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R2Shapes/R2Shapes.h"
#include "RNMath/RNMath.h"
#include "hdf5.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static const char *input_depth_filename = NULL;
static const char *input_duv_filename = NULL;
static const char *input_normals_filename = NULL;
static const char *input_nx_filename = NULL;
static const char *input_ny_filename = NULL;
static const char *input_nz_filename = NULL;
static const char *input_du_filename = NULL;
static const char *input_dv_filename = NULL;
static const char *input_inertia_depth_filename = NULL;
static const char *input_inertia_weight_filename = NULL;
static const char *input_xsmoothness_weight_filename = NULL;
static const char *input_ysmoothness_weight_filename = NULL;
static const char *input_normal_weight_filename = NULL;
static const char *input_tangent_weight_filename = NULL;
static const char *input_derivative_weight_filename = NULL;
static const char *input_range_weight_filename = NULL;
static const char *output_depth_filename = NULL;
static const char *output_plot_filename = NULL;
static const char *true_depth_filename = NULL;
static double minimum_depth = 0.05;
static double maximum_depth = 20;
static double png_depth_scale = 4000;
static double inertia_weight = 0;
static double smoothness_weight = 1E-3;
static double derivative_weight = 1;
static double normal_weight = 0;
static double tangent_weight = 1;
static double range_weight = 0;
static int normalize_tangent_vectors = 0;
static int xres = 0;
static int yres = 0;
static int solver = RN_CSPARSE_SOLVER;
static R3Matrix camera_intrinsics(0, 0, 0, 0, 0, 0, 0, 0, 1); 
static double gravity_vector_in_camera_coordinates[3] = { 0, 0, -1 };
static double plot_max_value = 1;
static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Input images
////////////////////////////////////////////////////////////////////////

static R2Grid *input_depth_image = NULL;
static R2Grid *input_normals_images[3] = { NULL, NULL, NULL };
static R2Grid *input_duv_images[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
static R2Grid *input_inertia_depth_image = NULL;
static R2Grid *input_inertia_weight_image = NULL;
static R2Grid *input_smoothness_weight_images[2] = { NULL, NULL };
static R2Grid *input_normal_weight_image = NULL;
static R2Grid *input_tangent_weight_image = NULL;
static R2Grid *input_derivative_weight_image = NULL;
static R2Grid *input_range_weight_image = NULL;
static R2Grid *true_depth_image = NULL;



////////////////////////////////////////////////////////////////////////
// Output images
////////////////////////////////////////////////////////////////////////

static R2Grid *output_depth_image = NULL;



////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static int
ResampleImage(R2Grid *image, int xres, int yres)
{
  // Check resolution
  if ((image->XResolution() == xres) && (image->YResolution() == yres)) return 1;

  // Compute scale factors
  RNScalar xscale = (RNScalar) image->XResolution() / (RNScalar) xres;
  RNScalar yscale = (RNScalar) image->YResolution() / (RNScalar) yres;

  // Resample grid values at new resolution
  R2Grid copy = *image;
  *image = R2Grid(xres, yres);
  for (int i = 0; i < xres; i++) {
    int ix = (int) (i*xscale + 0.5);
    if (ix > copy.XResolution()-1) ix = copy.XResolution()-1;
    for (int j = 0; j < yres; j++) {
      int iy = (int) (j*yscale + 0.5);
      if (iy > copy.YResolution()-1) iy = copy.YResolution()-1;
      RNScalar grid_value = copy.GridValue(ix, iy);
      if (grid_value < 1) RNBreakDebug();
      image->SetGridValue(i, j, grid_value);
    }
  }
  
  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Input and output functions
////////////////////////////////////////////////////////////////////////

static R2Grid *
ReadImage(const char *filename, RNScalar png_scale, RNScalar png_offset, int print_verbose = 0)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate a grid
  R2Grid *grid = new R2Grid();
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid for %s\n", filename);
    exit(-1);
    return NULL;
  }

  // Read grid
  if (!grid->Read(filename)) {
    fprintf(stderr, "Unable to read grid file %s\n", filename);
    exit(-1);
    return NULL;
  }

  // Process png file
  if (strstr(filename, ".png")) {
    if (png_offset != 0) grid->Subtract(png_offset);
    if (png_scale != 1) grid->Divide(png_scale);
  }

  // Remember grid resolution
  if (xres == 0) xres = grid->XResolution();
  if (yres == 0) yres = grid->YResolution();

  // Update default camera intrinsics
  if (camera_intrinsics[0][2] == 0) camera_intrinsics[0][2] = 0.5*xres;
  if (camera_intrinsics[1][2] == 0) camera_intrinsics[1][2] = 0.5*yres;

  // Print statistics
  if (print_verbose) {
    printf("Read image from %s\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", grid->XResolution(), grid->YResolution());
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return grid
  return grid;
}



static int
ReadH5(const char *filename, R2Grid **grids, unsigned int ngrids,
       int print_verbose = 0, const char *dataset_name = "/result")
{
  // Start statistics
  if (ngrids == 0) return 1;
  RNTime start_time;
  start_time.Read();

  // Open HDF5 file 
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t dataset_id = H5Dopen (file_id , dataset_name, H5P_DEFAULT );
  hid_t file_dataspace_id = H5Dget_space(dataset_id);

  // Check number of dimensions
  unsigned int rank = H5Sget_simple_extent_ndims (file_dataspace_id);
  if (rank < 3) {
    fprintf(stderr, "There should be at least 3 dimensions in hdf5 file %s\n", filename);
    exit(-1);
    return 0;
  }

  // Get dimensions
  hsize_t *dims = new hsize_t [ rank ];
  unsigned int ndims = H5Sget_simple_extent_dims ( file_dataspace_id , dims , NULL );
  if (ndims != rank ) {
    fprintf (stderr , "Mismatching number of dimensions in %s: %d vs. %d\n ", filename, rank , ndims);
    exit(-1);
    return 0;
  }

  // Get number of elements
  hssize_t num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
  if (num_elem != (hssize_t) (dims[ndims-3]*dims[ndims-2]*dims[ndims-1])) {
    fprintf (stderr, "Mismatching number of elements in %s\n ", filename);
    exit(-1);
    return 0;
  }

  // Allocate matrix
  float *A = new float [ num_elem ];
  if (!A) {
    fprintf(stderr, "Unable to allocate matrix for %s\n", filename);
    exit(-1);
    return 0;
  }

  // Create dataspace
  hid_t dataspace_id = H5Screate_simple(rank, dims, NULL);

  // Read matrix data from file
  int status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, dataspace_id, file_dataspace_id, H5P_DEFAULT, A);
  if (status < 0) {
    fprintf (stderr, "Unable to read data from %s\n ", filename);
    exit(-1);
    return 0;
  }

  // Fill images
  float *a = A;
  for (unsigned int i = 0; i < dims[ndims-3]; i++) {
    grids[i] = new R2Grid(dims[ndims-1], dims[ndims-2]);
    for (unsigned int k = 0; k < dims[ndims-2]; k++) {
      for (unsigned int j = 0; j < dims[ndims-1]; j++) {
        grids[i]->SetGridValue(j, dims[ndims-2]-1-k, *(a++));
      }
    }
  }

  // Release resources and close files
  status = H5Dclose ( dataset_id );
  status = H5Sclose ( dataspace_id );
  status = H5Sclose ( file_dataspace_id );
  status = H5Fclose ( file_id );

  // Free temporary data
  delete [] dims;
  delete [] A;

  // Print statistics
  if (print_verbose) {
    printf("Read images from %s\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", grids[0]->XResolution(), grids[0]->YResolution());
    printf("  Cardinality = %d\n", grids[0]->Cardinality());
    RNInterval grid_range = grids[0]->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grids[0]->L1Norm());
    printf("  L2Norm = %g\n", grids[0]->L2Norm());
    fflush(stdout);
  }

  // Return succes
  return 1;
}



static int
ReadInputs(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;
  
  // Read input depth image
  if (input_depth_filename) {
    input_depth_image = ReadImage(input_depth_filename, png_depth_scale, 0, print_verbose);
    input_depth_image->Substitute(0, R2_GRID_UNKNOWN_VALUE);
    ResampleImage(input_depth_image, xres, yres);
    if (print_debug) input_depth_image->WriteFile("id.pfm");
  }

  // Read true depth image
  if (true_depth_filename) {
    true_depth_image = ReadImage(true_depth_filename, png_depth_scale, 0, print_verbose);
    true_depth_image->Substitute(0, R2_GRID_UNKNOWN_VALUE);
    ResampleImage(true_depth_image, xres, yres);
    if (print_debug) true_depth_image->WriteFile("td.pfm");
  }

  // Read duv images
  if (input_duv_filename) {
    ReadH5(input_duv_filename, input_duv_images, 8, print_verbose);
    if (input_duv_images[0]->XResolution() != xres) RNAbort("Mismatching resolution in %s", input_duv_filename);
    if (input_duv_images[0]->YResolution() != yres) RNAbort("Mismatching resolution in %s", input_duv_filename);
    for (int i = 0; i < 8; i++)  input_duv_images[i]->Threshold(-20, R2_GRID_UNKNOWN_VALUE, R2_GRID_KEEP_VALUE);
    for (int i = 0; i < 8; i++)  input_duv_images[i]->Threshold(20, R2_GRID_KEEP_VALUE, R2_GRID_UNKNOWN_VALUE);
    char buffer[1024];
    for (int i = 0; i < 8; i++) { sprintf(buffer, "duv%d.pfm", i); input_duv_images[i]->WriteFile(buffer); }
    count += 2;
    if (print_debug) {
      input_duv_images[0]->WriteFile("du.pfm");
      input_duv_images[1]->WriteFile("dv.pfm");
    }
  }

  // Read normal images
  if (input_normals_filename) {
    ReadH5(input_normals_filename, input_normals_images, 3, print_verbose);
    if (input_normals_images[0]->XResolution() != xres) RNAbort("Mismatching resolution in %s", input_normals_filename);
    if (input_normals_images[0]->YResolution() != yres) RNAbort("Mismatching resolution in %s", input_normals_filename);
    R2Grid *swap = input_normals_images[1]; input_normals_images[1] = input_normals_images[2]; input_normals_images[2] = swap;
    input_normals_images[2]->Negate();
    count += 3;
    if (print_debug) {
      input_normals_images[0]->WriteFile("nx.pfm");
      input_normals_images[1]->WriteFile("ny.pfm");
      input_normals_images[2]->WriteFile("nz.pfm");
    }
  }

  // Read du image
  if (input_du_filename && !input_duv_images[0]) {
    input_duv_images[0] = ReadImage(input_du_filename, png_depth_scale, 32768, print_verbose);
    if (input_duv_images[0]->XResolution() != xres) RNAbort("Mismatching resolution in %s", input_du_filename);
    if (input_duv_images[0]->YResolution() != yres) RNAbort("Mismatching resolution in %s", input_du_filename);
    count++;
  }

  // Read dv image
  if (input_dv_filename && !input_duv_images[1]) {
    input_duv_images[1] = ReadImage(input_dv_filename, png_depth_scale, 32768, print_verbose);
    if (input_duv_images[1]->XResolution() != xres) RNAbort("Mismatching resolution in %s", input_dv_filename);
    if (input_duv_images[1]->YResolution() != yres) RNAbort("Mismatching resolution in %s", input_dv_filename);
    count++;
  }

  // Read nx image
  if (input_nx_filename && !input_normals_images[0]) {
    input_normals_images[0] = ReadImage(input_nx_filename, 32768, 32768, print_verbose);
    ResampleImage(input_normals_images[0], xres, yres);
    count++;
  }

  // Read ny image
  if (input_ny_filename && !input_normals_images[1]) {
    input_normals_images[1] = ReadImage(input_ny_filename, 32768, 32768, print_verbose);
    ResampleImage(input_normals_images[1], xres, yres);
    count++;
  }

  // Read nx image
  if (input_nz_filename && !input_normals_images[2]) {
    input_normals_images[2] = ReadImage(input_nz_filename, 32768, 32768, print_verbose);
    ResampleImage(input_normals_images[2], xres, yres);
    count++;
  }

  // Read inertia depth image
  if (input_inertia_depth_filename && !input_inertia_depth_image) {
    input_inertia_depth_image = ReadImage(input_inertia_depth_filename, png_depth_scale, 0, print_verbose);
    input_inertia_depth_image->Resample(xres, yres);
    count++;
  }

  // Read inertia weight image
  if (input_inertia_weight_filename && !input_inertia_weight_image) {
    input_inertia_weight_image = ReadImage(input_inertia_weight_filename, 1000, 0, print_verbose);
    input_inertia_weight_image->Resample(xres, yres);
    count++;
  }

  // Read xsmoothness_weight image
  if (input_xsmoothness_weight_filename && !input_smoothness_weight_images[0]) {
    input_smoothness_weight_images[0] = ReadImage(input_xsmoothness_weight_filename, 1000, 0, print_verbose);
    input_smoothness_weight_images[0]->Resample(xres, yres);
    count++;
  }

  // Read ysmoothness_weight image
  if (input_ysmoothness_weight_filename && !input_smoothness_weight_images[1]) {
    input_smoothness_weight_images[1] = ReadImage(input_ysmoothness_weight_filename, 1000, 0, print_verbose);
    input_smoothness_weight_images[1]->Resample(xres, yres);
    count++;
  }

  // Read normal image
  if (input_normal_weight_filename && !input_normal_weight_image) {
    input_normal_weight_image = ReadImage(input_normal_weight_filename, 1000, 0, print_verbose);
    input_normal_weight_image->Resample(xres, yres);
    count++;
  }

  // Read tangent image
  if (input_tangent_weight_filename && !input_tangent_weight_image) {
    input_tangent_weight_image = ReadImage(input_tangent_weight_filename, 1000, 0, print_verbose);
    input_tangent_weight_image->Resample(xres, yres);
    count++;
  }

  // Read derivative weight image
  if (input_derivative_weight_filename && !input_derivative_weight_image) {
    input_derivative_weight_image = ReadImage(input_derivative_weight_filename, 1000, 0, print_verbose);
    input_derivative_weight_image->Resample(xres, yres);
    count++;
  }

  // Read range weight image
  if (input_range_weight_filename && !input_range_weight_image) {
    input_range_weight_image = ReadImage(input_range_weight_filename, 1000, 0, print_verbose);
    input_range_weight_image->Resample(xres, yres);
    count++;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read images ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", xres, yres);
    printf("  # Images = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int 
WriteImage(R2Grid *grid, const char *filename, RNScalar png_scale, RNScalar png_offset, int print_verbose = 0)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Copy grid so can be processed
  R2Grid tmp = *grid;
  
  // Process png file
  if (strstr(filename, ".png")) {
    if (png_scale != 1) tmp.Multiply(png_scale);
    if (png_offset != 0) tmp.Add(png_offset);
    tmp.Threshold(0, 0, R2_GRID_KEEP_VALUE);
    tmp.Threshold(65535, R2_GRID_KEEP_VALUE, 65535);
  }

  // Write grid
  if (!tmp.Write(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Wrote image to %s\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", grid->XResolution(), grid->YResolution());
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return success
  return 1;
}



#if 0
static int
WriteErrorImages(R2Grid *output_depth_image, R2Grid *true_depth_image)
{
  // Check camera intrinsics
  if (RNIsZero(camera_intrinsics[0][0]) || RNIsZero(camera_intrinsics[1][1])) {  
    fprintf(stderr, "You must provide camera intrinsics to create normal image\n");
    return 0;
  }

  // Create normal image
  R2Grid normal_images[3];
  normal_images[0] = R2Grid(output_depth_image->XResolution(), output_depth_image->YResolution(), 3);
  normal_images[1] = R2Grid(output_depth_image->XResolution(), output_depth_image->YResolution(), 3);
  normal_images[2] = R2Grid(output_depth_image->XResolution(), output_depth_image->YResolution(), 3);
  for (int ix = 0; ix < output_depth_image->XResolution(); ix++) {
    for (int iy = 0; iy < output_depth_image->YResolution(); iy++) {
    }
  }
}
#endif



static int
WriteErrorPlot(R2Grid *output_depth_image, R2Grid *true_depth_image, const char *output_plot_filename, RNScalar plot_max_value, int print_verbose)
{
  // Check inputs
  if (!true_depth_image) return 0;
  if (!output_depth_image) return 0;
  if (!output_plot_filename) return 0;
  if (plot_max_value <= 0) return 0;

  // Compute difference between true and output images
  R2Grid image1 = *true_depth_image;
  R2Grid image2 = *output_depth_image;
  image2.Mask(image1);
  RNScalar median1 = image1.Median();
  RNScalar median2 = image2.Median();
  image1.Subtract(median1);
  image2.Subtract(median2);
  R2Grid difference_image(image2);
  difference_image.Subtract(image1);

  // Create array of depth differences
  int nvalues = 0;
  RNScalar sum = 0;
  RNScalar *values = new RNScalar [ difference_image.NEntries() ];
  for (int i = 0; i < difference_image.NEntries(); i++) {
    RNScalar value = difference_image.GridValue(i);
    if (value == R2_GRID_UNKNOWN_VALUE) continue;
    values[nvalues++] = fabs(value);
    sum += fabs(value);
  }

  // Sort array of depth differences 
  if (nvalues == 0) return 0;
  qsort(values, nvalues, sizeof(RNScalar), RNCompareScalars);

  // Open plot file
  FILE *fp = fopen(output_plot_filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open plot file %s\n", output_plot_filename);
    return 0;
  }

  // Write plot file
  int nbins = 100;
  int count = 0;
  fprintf(fp, "0 0\n");
  for (int k = 0; k < nbins; k++) {
    RNScalar max_bin_value = (k+1)*plot_max_value/nbins;
    while ((count < nvalues) && (values[count] <= max_bin_value)) count++; 
    fprintf(fp, "%g %g\n", max_bin_value, (RNScalar) count / (RNScalar) nvalues);
  }

  // Close plot file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote error plot to %s ...\n", output_plot_filename);
    printf("  Median values = %g %g ( %g )\n", median1, median2, fabs(median1 - median2));
    printf("  Min error = %g\n", values[0]);
    printf("  Max error = %g\n", values[nvalues-1]);
    printf("  Mean error = %g\n", sum / nvalues);
    printf("  Median error = %g\n", values[nvalues/2]);
  }

  // Delete values
  delete [] values;

  // Return success
  return 1;
}
  


static int
WriteOutputs(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write output depth image
  if (!WriteImage(output_depth_image, output_depth_filename, png_depth_scale, 0, print_verbose)) return 0;

  // Write error plot
  if (!WriteErrorPlot(output_depth_image, true_depth_image, output_plot_filename, plot_max_value, print_verbose)) return 0;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Equation definition functions
////////////////////////////////////////////////////////////////////////

static int
CreateSmoothnessEquations(RNSystemOfEquations& equations)
{
  // Check smoothness weight
  if ((smoothness_weight == 0) && !input_smoothness_weight_images[0] && !input_smoothness_weight_images[1]) return 1;
  
  // Create smoothness equations
  int count = 0;
  for (int iy = 0; iy < yres; iy++) {
    for (int ix = 0; ix < xres; ix++) {
      // Check if pixel is in a hole
      if (FALSE) continue;

      // Add smoothness equations
      if (ix > 0) {
        RNScalar w = smoothness_weight;
        if (input_smoothness_weight_images[0]) w *= input_smoothness_weight_images[0]->GridValue(ix-1, iy);
        if (w > 0) {
          RNPolynomial *e = new RNPolynomial();
          e->AddTerm(-1.0, (iy)*xres+(ix),   1.0);
          e->AddTerm( 1.0, (iy)*xres+(ix-1), 1.0);
          e->Multiply(w);
          equations.InsertEquation(e);
          count++;
        }
      }
      if (ix < xres-1) {
        RNScalar w = smoothness_weight;
        if (input_smoothness_weight_images[0]) w *= input_smoothness_weight_images[0]->GridValue(ix, iy);
        if (w > 0) {
          RNPolynomial *e = new RNPolynomial();
          e->AddTerm(-1.0, (iy)*xres+(ix),   1.0);
          e->AddTerm( 1.0, (iy)*xres+(ix+1), 1.0);
          e->Multiply(w);
          equations.InsertEquation(e);
          count++;
        }
      }
      if (iy > 0) {
        RNScalar w = smoothness_weight;
        if (input_smoothness_weight_images[1]) w *= input_smoothness_weight_images[1]->GridValue(ix, iy-1);
        if (w > 0) {
          RNPolynomial *e = new RNPolynomial();
          e->AddTerm(-1.0, (iy)  *xres+(ix),   1.0);
          e->AddTerm( 1.0, (iy-1)*xres+(ix), 1.0);
          e->Multiply(w);
          equations.InsertEquation(e);
          count++;
        }
      }
      if (iy < yres-1) {
        RNScalar w = smoothness_weight;
        if (input_smoothness_weight_images[1]) w *= input_smoothness_weight_images[1]->GridValue(ix, iy);
        if (w > 0) {
          RNPolynomial *e = new RNPolynomial();
          e->AddTerm(-1.0, (iy)  *xres+(ix),   1.0);
          e->AddTerm( 1.0, (iy+1)*xres+(ix), 1.0);
          e->Multiply(w);
          equations.InsertEquation(e);
          count++;
        }
      }
    }
  }

  // Return success
  return 1;
}



static int
CreateInertiaEquations(RNSystemOfEquations& equations)
{
  RNBoolean found = FALSE;

  // Get target depth image
  R2Grid *depth_image = input_inertia_depth_image;
  if (!depth_image) depth_image = input_depth_image;

  // Create equations to preserve depths
  if (depth_image && (inertia_weight > 0) && input_inertia_weight_image) {
    // Preserve depth in given depth image
    for (int i = 0; i < xres*yres; i++) {
      RNScalar w = input_inertia_weight_image->GridValue(i);
      if ((w <= 0) || (w == R2_GRID_UNKNOWN_VALUE)) continue;
      RNScalar d = depth_image->GridValue(i);
      if ((d == 0) || (d == R2_GRID_UNKNOWN_VALUE)) continue;
      RNPolynomial *e = new RNPolynomial(1.0, i, 1.0);
      e->Subtract(d);
      e->Multiply(w * inertia_weight);
      equations.InsertEquation(e);
      found = TRUE;
    }
  }
  else if (depth_image && (inertia_weight > 0)) {
    // Preserve depth in given depth image
    for (int i = 0; i < xres*yres; i++) {
      RNScalar d = depth_image->GridValue(i);
      if ((d == 0) || (d == R2_GRID_UNKNOWN_VALUE)) continue;
      RNPolynomial *e = new RNPolynomial(1.0, i, 1.0);
      e->Subtract(d);
      e->Multiply(inertia_weight);
      equations.InsertEquation(e);
      found = TRUE;
    }
  }
  else if (depth_image) {
    // Set depth of one pixel
    int ix, iy, n = xres*yres;
    for (int i = 0; i < n; i++) {
      int index = RNRandomScalar() * n;
      depth_image->IndexToIndices(index, ix, iy);
      if ((ix < xres/4) || (ix > 3*xres/4)) continue;
      if ((iy < yres/4) || (iy > 3*yres/4)) continue;
      RNScalar d = depth_image->GridValue(index);
      if ((d != 0) && (d != R2_GRID_UNKNOWN_VALUE)) {
        RNPolynomial *e = new RNPolynomial(1.0, index, 1.0);
        e->Subtract(d);
        e->Multiply(1000);
        equations.InsertEquation(e);
        found = TRUE;
        break;
      }
    }
  }

  // Last resort
  if (!found) {
    // Set depth of middle pixel to zero
    RNPolynomial *e = new RNPolynomial(1.0, (yres/2)*xres+(xres/2), 1.0);
    e->Multiply(1000);
    equations.InsertEquation(e);
  }

  // Return success
  return 1;
}


  
static int
CreateDUVEquations(RNSystemOfEquations& equations)
{
  // Check images/weight
  if (!input_duv_images[0] || (derivative_weight == 0)) return 1;
  
  // Create derivative equations
  for (int i = 0; i < 8; i++) {
    if (!input_duv_images[i]) continue;
    int sx = 0, sy = 0;
    if (i == 0)      { sx = -1; sy =  1; }
    else if (i == 1) { sx =  0; sy =  1; }
    else if (i == 2) { sx =  1; sy =  1; }
    else if (i == 3) { sx = -1; sy =  0; }
    else if (i == 4) { sx =  1; sy =  0; }
    else if (i == 5) { sx = -1; sy = -1; }
    else if (i == 6) { sx =  0; sy = -1; }
    else if (i == 7) { sx =  1; sy = -1; }
    for (int iy = 0; iy < yres; iy++) {
      int ny = iy+sy;
      if ((ny < 0) || (ny >= input_duv_images[i]->YResolution())) continue;
      for (int ix = 0; ix < xres-1; ix++) {
        int nx = ix+sx;
        if ((nx < 0) || (nx >= input_duv_images[i]->XResolution())) continue;
        RNScalar d = input_duv_images[i]->GridValue(ix, iy);
        if (d == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar w = derivative_weight;
        if (input_derivative_weight_image) w *= input_derivative_weight_image->GridValue(ix, iy);
        if (w == 0) continue;
        RNPolynomial *e = new RNPolynomial();
        e->AddTerm(1.0, (iy)*xres+(ix), 1.0);
        e->AddTerm(-1.0,(ny)*xres+(nx), 1.0);
        e->Subtract(d);
        e->Multiply(w);
        equations.InsertEquation(e);
      }
    }
  }

  // Return success
  return 1;
}



#if 0
static int
CreateLUVEquations(RNSystemOfEquations& equations)
{
  // Create lu equations
  if (input_lu_image && (lu_weight > 0)) {
    for (int iy = 0; iy < yres; iy++) {
      for (int ix = 1; ix < xres-1; ix++) {
        RNScalar lu = input_lu_image->GridValue(ix, iy);
        if (lu == R2_GRID_UNKNOWN_VALUE) continue;
        RNPolynomial *e = new RNPolynomial();
        e->AddTerm(-0.5, (iy)*xres+(ix-1), 1.0);
        e->AddTerm( 1.0, (iy)*xres+(ix),   1.0);
        e->AddTerm(-0.5, (iy)*xres+(ix+1), 1.0);
        e->Subtract(lu);
        e->Multiply(lu_weight);
        equations.InsertEquation(e);
      }
    }
  }
  
  // Create lv equations
  if (input_lv_image && (lv_weight > 0)) {
    for (int iy = 1; iy < yres-1; iy++) {
      for (int ix = 0; ix < xres; ix++) {
        RNScalar lv = input_lv_image->GridValue(ix, iy);
        if (lv == R2_GRID_UNKNOWN_VALUE) continue;
        RNPolynomial *e = new RNPolynomial();
        e->AddTerm(-0.5, (iy-1)*xres+(ix), 1.0);
        e->AddTerm( 1.0, (iy)  *xres+(ix), 1.0);
        e->AddTerm(-0.5, (iy+1)*xres+(ix), 1.0);
        e->Subtract(lv);
        e->Multiply(lv_weight);
        equations.InsertEquation(e);
      }
    }
  }
  
  // Create luv equations
  if (input_luv_image && (luv_weight > 0)) {
    for (int iy = 1; iy < yres-1; iy++) {
      for (int ix = 1; ix < xres-1; ix++) {
        RNScalar luv = input_luv_image->GridValue(ix, iy);
        if (luv == R2_GRID_UNKNOWN_VALUE) continue;
        RNPolynomial *e = new RNPolynomial();
        RNScalar weight = 0;
        if (ix > 0) {
          if (iy > 0) {
            e->AddTerm(-0.0625, (iy-1)*xres+(ix-1), 1.0);
            weight += 0.0625;
          }
          if (TRUE) {
            e->AddTerm(-0.125, (iy)*xres+(ix-1), 1.0);
            weight += 0.125;
          }
          if (iy < yres-1) {
            e->AddTerm(-0.0625, (iy+1)*xres+(ix-1), 1.0);
            weight += 0.0625;
          }
        }
        if (TRUE) {
          if (iy > 0) {
            e->AddTerm(-0.125, (iy-1)*xres+(ix), 1.0);
            weight += 0.125;
          }
          if (TRUE) {
            e->AddTerm(-0.25, (iy)*xres+(ix), 1.0);
            weight += 0.25;
          }
          if (iy < yres-1) {
            e->AddTerm(-0.125, (iy+1)*xres+(ix), 1.0);
            weight += 0.125;
          }
        }
        if (ix < xres-1) {
          if (iy > 0) {
            e->AddTerm(-0.0625, (iy-1)*xres+(ix+1), 1.0);
            weight += 0.0625;
          }
          if (TRUE) {
            e->AddTerm(-0.125, (iy)*xres+(ix+1), 1.0);
            weight += 0.125;
          }
          if (iy < yres-1) {
            e->AddTerm(-0.0625, (iy+1)*xres+(ix+1), 1.0);
            weight += 0.0625;
          }
        }
        e->AddTerm(weight, (iy)*xres+(ix), 1.0);
        e->Subtract(weight * luv);
        e->Multiply(luv_weight);
        equations.InsertEquation(e);
      }
    }
  }

  // Return success
  return 1;
}
#endif



#if 0
static int
CreateDXYEquations(RNSystemOfEquations& equations)
{
  // Create dx equations
  if (input_dx_image && (dx_weight > 0)) {
    for (int iy = 0; iy < yres; iy++) {
      for (int ix = 0; ix < xres-1; ix++) {
        RNScalar dx = input_dx_image->GridValue(ix, iy);
        if (dx == R2_GRID_UNKNOWN_VALUE) continue;
        RNPolynomial *d = new RNPolynomial();
        d->AddTerm(1.0, (iy)*xres+(ix),   1.0);
        RNPolynomial *d0 = new RNPolynomial();
        d0->AddTerm(1.0, (iy)*xres+(ix),   1.0);
        RNPolynomial *d1 = new RNPolynomial();
        d1->AddTerm(1.0, (iy)*xres+(ix+1),   1.0);
        RNAlgebraic *e = new RNAlgebraic(RN_SUBTRACT_OPERATION, d1, d0);
        e = new RNAlgebraic(RN_DIVIDE_OPERATION, e, d);
        e = new RNAlgebraic(RN_SUBTRACT_OPERATION, e, dx);
        e = new RNAlgebraic(RN_MULTIPLY_OPERATION, e, dx_weight);
        equations.InsertEquation(e);
      }
    }
  }
  
  // Create dy equations
  if (input_dy_image && (dy_weight > 0)) {
    for (int iy = 0; iy < yres-1; iy++) {
      for (int ix = 0; ix < xres; ix++) {
        RNScalar dy = input_dy_image->GridValue(ix, iy);
        if (dy == R2_GRID_UNKNOWN_VALUE) continue;
        RNPolynomial *d = new RNPolynomial();
        d->AddTerm(1.0, (iy)*xres+(ix),   1.0);
        RNPolynomial *d0 = new RNPolynomial();
        d0->AddTerm(1.0, (iy)*xres+(ix),   1.0);
        RNPolynomial *d1 = new RNPolynomial();
        d1->AddTerm(1.0, (iy+1)*xres+(ix),   1.0);
        RNAlgebraic *e = new RNAlgebraic(RN_SUBTRACT_OPERATION, d1, d0);
        e = new RNAlgebraic(RN_DIVIDE_OPERATION, e, d);
        e = new RNAlgebraic(RN_SUBTRACT_OPERATION, e, dy);
        e = new RNAlgebraic(RN_MULTIPLY_OPERATION, e, dy_weight);
        equations.InsertEquation(e);
      }
    }
  }

  // Return success
  return 1;
}
#endif



static int
CreateNormalEquations(RNSystemOfEquations& equations)
{
  // Create linear normal equations
  if (input_normals_images[0] && input_normals_images[1] && input_normals_images[2] && (normal_weight > 0)) {
    // Check camera intrinsics
    if (RNIsZero(camera_intrinsics[0][0]) || RNIsZero(camera_intrinsics[1][1])) {  
      fprintf(stderr, "You must provide camera intrinsics to create normal equations\n");
      return 0;
    }

    // Create normal equations
    for (int iy = 0; iy < yres; iy++) {
      for (int ix = 0; ix < xres; ix++) {
        RNScalar input_nx = input_normals_images[0]->GridValue(ix, iy);
        if (input_nx == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar input_ny = input_normals_images[1]->GridValue(ix, iy);
        if (input_ny == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar input_nz = input_normals_images[2]->GridValue(ix, iy);
        if (input_nz == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar w = normal_weight;
        if (!normalize_tangent_vectors) w *= camera_intrinsics[0][0];
        if (input_normal_weight_image) w *= input_normal_weight_image->GridValue(ix, iy);
        if (w == 0) continue;

        // Check normal direction
        if (RNIsNegativeOrZero(input_nz)) continue;

        // Consider 4 directions
        for (int dir = 0; dir < 4; dir++) {
          // Get image coordinates of adjacent point
          int ixA , iyA, ixB, iyB;
          if (dir == 0) { ixA = ix+1; iyA = iy; ixB = ix; iyB = iy+1; }
          else if (dir == 1) { ixA = ix; iyA = iy+1; ixB = ix-1; iyB = iy; }
          else if (dir == 2) { ixA = ix-1; iyA = iy; ixB = ix; iyB = iy-1; }
          else { ixA = ix; iyA = iy-1; ixB = ix+1; iyB = iy; }
          if ((ixA < 0) || (ixA >= xres)) continue;
          if ((iyA < 0) || (iyA >= yres)) continue;
          if ((ixB < 0) || (ixB >= xres)) continue;
          if ((iyB < 0) || (iyB >= yres)) continue;

          // Compute depths
          RNPolynomial d(1.0, (iy)*xres+(ix), 1.0);
          RNPolynomial dA(1.0, (iyA)*xres+(ixA), 1.0);
          RNPolynomial dB(1.0, (iyB)*xres+(ixB), 1.0);

          // Compute camera coordinates
          RNPolynomial x = d * ((ix - camera_intrinsics[0][2]) / camera_intrinsics[0][0]);
          RNPolynomial y = d * ((iy - camera_intrinsics[1][2]) / camera_intrinsics[1][1]);
          RNPolynomial xA = dA * ((ixA - camera_intrinsics[0][2]) / camera_intrinsics[0][0]);
          RNPolynomial yA = dA * ((iyA - camera_intrinsics[1][2]) / camera_intrinsics[1][1]);
          RNPolynomial xB = dB * ((ixB - camera_intrinsics[0][2]) / camera_intrinsics[0][0]);
          RNPolynomial yB = dB * ((iyB - camera_intrinsics[1][2]) / camera_intrinsics[1][1]);

          // Compute tangent vector
          RNAlgebraic *dxA = new RNAlgebraic(xA - x, 0);
          RNAlgebraic *dyA = new RNAlgebraic(yA - y, 0);
          RNAlgebraic *dzA = new RNAlgebraic(d - dA, 0);
          RNAlgebraic *dxB = new RNAlgebraic(xB - x, 0);
          RNAlgebraic *dyB = new RNAlgebraic(yB - y, 0);
          RNAlgebraic *dzB = new RNAlgebraic(d - dB, 0);

          // Compute cross product of vA and vB
          RNAlgebraic *cx1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNAlgebraic(*dyA), new RNAlgebraic(*dzB));
          RNAlgebraic *cx2 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNAlgebraic(*dzA), new RNAlgebraic(*dyB));
          RNAlgebraic *cx = new RNAlgebraic(RN_SUBTRACT_OPERATION, cx1, cx2);
          RNAlgebraic *cy1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, dzA, new RNAlgebraic(*dxB));
          RNAlgebraic *cy2 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNAlgebraic(*dxA), dzB);
          RNAlgebraic *cy = new RNAlgebraic(RN_SUBTRACT_OPERATION, cy1, cy2);
          RNAlgebraic *cz1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, dxA, dyB);
          RNAlgebraic *cz2 = new RNAlgebraic(RN_MULTIPLY_OPERATION, dyA, dxB);
          RNAlgebraic *cz = new RNAlgebraic(RN_SUBTRACT_OPERATION, cz1, cz2);

          // Compute normal
          RNAlgebraic *scx = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*cx), 2);
          RNAlgebraic *scy = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*cy), 2);
          RNAlgebraic *scz = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*cz), 2);
          RNAlgebraic *slen = new RNAlgebraic(RN_ADD_OPERATION, scx, scy);
          slen = new RNAlgebraic(RN_ADD_OPERATION, slen, scz);
          RNAlgebraic *len = new RNAlgebraic(RN_POW_OPERATION, slen, 0.5);
          RNAlgebraic *nx = new RNAlgebraic(RN_DIVIDE_OPERATION, cx, new RNAlgebraic(*len));
          RNAlgebraic *ny = new RNAlgebraic(RN_DIVIDE_OPERATION, cy, new RNAlgebraic(*len));
          RNAlgebraic *nz = new RNAlgebraic(RN_DIVIDE_OPERATION, cz, len);

          // Compute errors
          RNAlgebraic *ex = new RNAlgebraic(RN_SUBTRACT_OPERATION, nx, input_nx);
          RNAlgebraic *ey = new RNAlgebraic(RN_SUBTRACT_OPERATION, ny, input_ny);
          RNAlgebraic *ez = new RNAlgebraic(RN_SUBTRACT_OPERATION, nz, input_nz);

          // Multiply by weight
          ex = new RNAlgebraic(RN_MULTIPLY_OPERATION, ex, w);
          ey = new RNAlgebraic(RN_MULTIPLY_OPERATION, ey, w);
          ez = new RNAlgebraic(RN_MULTIPLY_OPERATION, ez, w);

          // Insert equations
          equations.InsertEquation(ex);
          equations.InsertEquation(ey);
          equations.InsertEquation(ez);
        }
      }
    }
  }

  // Return success
  return 1;
}



static int
CreateTangentEquations(RNSystemOfEquations& equations)
{
  // Create tangent-normal equations
  if (input_normals_images[0] && input_normals_images[1] && input_normals_images[2] && (tangent_weight > 0)) {
    // Check camera intrinsics
    if (RNIsZero(camera_intrinsics[0][0]) || RNIsZero(camera_intrinsics[1][1])) {  
      fprintf(stderr, "You must provide camera intrinsics to create normal equations\n");
      return 0;
    }

    // Create tangent equations
    for (int iy = 0; iy < yres; iy++) {
      for (int ix = 0; ix < xres; ix++) {
        RNScalar input_nx = input_normals_images[0]->GridValue(ix, iy);
        if (input_nx == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar input_ny = input_normals_images[1]->GridValue(ix, iy);
        if (input_ny == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar input_nz = input_normals_images[2]->GridValue(ix, iy);
        if (input_nz == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar w = tangent_weight;
        if (!normalize_tangent_vectors) w *= camera_intrinsics[0][0];
        if (input_tangent_weight_image) w *= input_tangent_weight_image->GridValue(ix, iy);
        if (w == 0) continue;

        // Check normal direction
        if (RNIsNegativeOrZero(input_nz)) continue;

        // Consider 4 tangent directions
        for (int dir = 0; dir < 4; dir++) {
          // Get image coordinates of adjacent point
          int ixA , iyA;
          if (dir == 0) { ixA = ix+1; iyA = iy; }
          else if (dir == 1) { ixA = ix; iyA = iy+1; }
          else if (dir == 2) { ixA = ix-1; iyA = iy;  }
          else { ixA = ix; iyA = iy-1; }
          if ((ixA < 0) || (ixA >= xres)) continue;
          if ((iyA < 0) || (iyA >= yres)) continue;

          // Compute depths
          RNPolynomial d(1.0, (iy)*xres+(ix), 1.0);
          RNPolynomial dA(1.0, (iyA)*xres+(ixA), 1.0);

          // Compute camera coordinates
          RNPolynomial x = d * ((ix - camera_intrinsics[0][2]) / camera_intrinsics[0][0]);
          RNPolynomial y = d * ((iy - camera_intrinsics[1][2]) / camera_intrinsics[1][1]);
          RNPolynomial xA = dA * ((ixA - camera_intrinsics[0][2]) / camera_intrinsics[0][0]);
          RNPolynomial yA = dA * ((iyA - camera_intrinsics[1][2]) / camera_intrinsics[1][1]);

          // Compute tangent vector
          RNAlgebraic *dx = new RNAlgebraic(xA - x, 0);
          RNAlgebraic *dy = new RNAlgebraic(yA - y, 0);
          RNAlgebraic *dz = new RNAlgebraic(d - dA, 0);

          // Normalize tangent vector
          if (normalize_tangent_vectors) {
            RNAlgebraic *ddx = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*dx), 2);
            RNAlgebraic *ddy = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*dy), 2);
            RNAlgebraic *ddz = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*dz), 2);
            RNAlgebraic *dd = new RNAlgebraic(RN_ADD_OPERATION, ddx, ddy);
            dd = new RNAlgebraic(RN_ADD_OPERATION, dd, ddz);
            RNAlgebraic *d = new RNAlgebraic(RN_POW_OPERATION, dd, 0.5);
            dx = new RNAlgebraic(RN_DIVIDE_OPERATION, dx, new RNAlgebraic(*d));
            dy = new RNAlgebraic(RN_DIVIDE_OPERATION, dy, new RNAlgebraic(*d));
            dz = new RNAlgebraic(RN_DIVIDE_OPERATION, dz, d);
          }

          // Compute dot product between tangent and normal
          RNAlgebraic *dotx = new RNAlgebraic(RN_MULTIPLY_OPERATION, dx, input_nx);
          RNAlgebraic *doty = new RNAlgebraic(RN_MULTIPLY_OPERATION, dy, input_ny);
          RNAlgebraic *dotz = new RNAlgebraic(RN_MULTIPLY_OPERATION, dz, input_nz);
          RNAlgebraic *dot = new RNAlgebraic(RN_ADD_OPERATION, dotx, doty);
          dot = new RNAlgebraic(RN_ADD_OPERATION, dot, dotz);

          // Add equation for error
          RNAlgebraic *e = new RNAlgebraic(RN_MULTIPLY_OPERATION, dot, w);
          equations.InsertEquation(e);
        }
      }
    }
  }

  // Return success
  return 1;
}



static int
CreateRangeEquations(RNSystemOfEquations& equations)
{
  // Check parameters
  if (range_weight == 0) return 1;
  
  // For now
  RNScalar minimum = 0.1;
  RNScalar maximum = 20.0;
  
  // Create equations that penalize values outside range
  for (int i = 0; i < xres*yres; i++) {
    // Compute weight
    RNScalar w = range_weight;
    if (input_range_weight_image) w *= input_range_weight_image->GridValue(i);
    if (w == 0) continue;

    // Add penalty function on low side of range
    RNPolynomial *d0 = new RNPolynomial(1.0, i, 1.0);
    RNAlgebraic *e0 = new RNAlgebraic(RN_ADD_OPERATION, d0, 1.0 - minimum);
    e0 = new RNAlgebraic(RN_POW_OPERATION, e0, -2);
    e0 = new RNAlgebraic(RN_MULTIPLY_OPERATION, e0, w);
    equations.InsertEquation(e0);

    // Add penalty function on high side of range
    RNPolynomial *d1 = new RNPolynomial(1.0, i, 1.0);
    RNAlgebraic *e1 = new RNAlgebraic(RN_SUBTRACT_OPERATION, maximum - 1.0, d1);
    e1 = new RNAlgebraic(RN_POW_OPERATION, e1, -2);
    e1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, e1, w);
    equations.InsertEquation(e1);
  }

  // Return success
  return 1;
}


  
/////////////////////////////////////////////////////////////////
// Core solver function
////////////////////////////////////////////////////////////////////////

static int
CreateDepthImage(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int n = xres*yres;
  if (n == 0) return 0;

  // Allocate variables
  double *x = new double [ n ];
  for (int i = 0; i < n; i++) x[i] = 1;

  // Create system of equations
  RNSystemOfEquations equations(n);

  // Add bounds
  for (int i = 0; i < n; i++) equations.SetLowerBound(i, minimum_depth);
  for (int i = 0; i < n; i++) equations.SetUpperBound(i, maximum_depth);

  // Add basic equations
  int equations_count =0;
  CreateInertiaEquations(equations);
  int inertia_equations_count = equations.NEquations() - equations_count;
  equations_count = equations.NEquations();
  CreateSmoothnessEquations(equations);
  int smoothness_equations_count = equations.NEquations() - equations_count;
  equations_count = equations.NEquations();

  // Solve for initial guess 
  equations.Minimize(x, RN_CSPARSE_SOLVER, 1E-3);

  // Print initial guess
  if (print_debug) {
    R2Grid tmp(xres, yres);
    for (int i = 0; i < n; i++) tmp.SetGridValue(i, x[i]);
    tmp.WriteFile("sd.pfm");
  }

  // Add more equations
  CreateDUVEquations(equations);
  int derivative_equations_count = equations.NEquations() - equations_count;
  equations_count = equations.NEquations();
  CreateNormalEquations(equations);
  int normal_equations_count = equations.NEquations() - equations_count;
  equations_count = equations.NEquations();
  CreateTangentEquations(equations);
  int tangent_equations_count = equations.NEquations() - equations_count;
  equations_count = equations.NEquations();
  CreateRangeEquations(equations);
  int range_equations_count = equations.NEquations() - equations_count;
  equations_count = equations.NEquations();

  // Log initial ssd
  RNScalar initial_ssd = equations.SumOfSquaredResiduals(x);
  if (print_debug) printf("A %d %d %g\n", equations.NVariables(), equations.NEquations(), initial_ssd);

  // Solve for depth
  if (!equations.Minimize(x, solver, 1E-3)) {
    fprintf(stderr, "Unable to minimize system of equations\n");
    delete [] x;
    return 0;
  }
  
  // Log final ssd
  RNScalar final_ssd = equations.SumOfSquaredResiduals(x);
  if (print_debug) printf("B %d %d %g\n", equations.NVariables(), equations.NEquations(), final_ssd);

  // Allocate output depth image
  output_depth_image = new R2Grid(xres, yres);
  if (!output_depth_image) {
    fprintf(stderr, "Unable to allocate output depth grid\n");
    delete [] x;
    return 0;
  }

  // Copy solution into output depth image
  for (int i = 0; i < n; i++) {
    output_depth_image->SetGridValue(i, x[i]);
  }

  // Print message
  if (print_verbose) {
    printf("Solved for depth image\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Variables = %d\n", equations.NVariables());
    printf("  # Equations = %d\n", equations.NEquations());
    printf("    Inertia Equations = %d\n", inertia_equations_count);
    printf("    Smoothness Equations = %d\n", smoothness_equations_count);
    printf("    Derivative Equations = %d\n", derivative_equations_count);
    printf("    Normal Equations = %d\n", normal_equations_count);
    printf("    Tangent Equations = %d\n", tangent_equations_count);
    printf("    Range Equations = %d\n", range_equations_count);
    printf("  Initial SSD = %g\n", initial_ssd);
    printf("  Final SSD = %g\n", final_ssd);
    fflush(stdout);
  }

  // Delete variables
  delete [] x;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Program argument parsing
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1; 
      else if (!strcmp(*argv, "-debug")) print_debug = 1; 
      else if (!strcmp(*argv, "-ceres")) solver = RN_CERES_SOLVER;
      else if (!strcmp(*argv, "-splm")) solver = RN_SPLM_SOLVER;
      else if (!strcmp(*argv, "-csparse")) solver = RN_CSPARSE_SOLVER;
      else if (!strcmp(*argv, "-input_normals")) { argc--; argv++; input_normals_filename = *argv; }
      else if (!strcmp(*argv, "-input_derivatives")) { argc--; argv++; input_duv_filename = *argv; }
      else if (!strcmp(*argv, "-input_duv")) { argc--; argv++; input_duv_filename = *argv; }
      else if (!strcmp(*argv, "-input_nx")) { argc--; argv++; input_nx_filename = *argv; }
      else if (!strcmp(*argv, "-input_ny")) { argc--; argv++; input_ny_filename = *argv; }
      else if (!strcmp(*argv, "-input_nz")) { argc--; argv++; input_nz_filename = *argv; }
      else if (!strcmp(*argv, "-input_du")) { argc--; argv++; input_du_filename = *argv; }
      else if (!strcmp(*argv, "-input_dv")) { argc--; argv++; input_dv_filename = *argv; }
      else if (!strcmp(*argv, "-input_inertia_depth")) { argc--; argv++; input_inertia_depth_filename = *argv; }
      else if (!strcmp(*argv, "-input_inertia_weight")) { argc--; argv++; input_inertia_weight_filename = *argv; }
      else if (!strcmp(*argv, "-input_xsmoothness_weight")) { argc--; argv++; input_xsmoothness_weight_filename = *argv; }
      else if (!strcmp(*argv, "-input_ysmoothness_weight")) { argc--; argv++; input_ysmoothness_weight_filename = *argv; }
      else if (!strcmp(*argv, "-input_normal_weight")) { argc--; argv++; input_normal_weight_filename = *argv; }
      else if (!strcmp(*argv, "-input_tangent_weight")) { argc--; argv++; input_tangent_weight_filename = *argv; }
      else if (!strcmp(*argv, "-input_derivative_weight")) { argc--; argv++; input_derivative_weight_filename = *argv; }
      else if (!strcmp(*argv, "-input_range_weight")) { argc--; argv++; input_range_weight_filename = *argv; }
      else if (!strcmp(*argv, "-output_plot")) { argc--; argv++; output_plot_filename = *argv; }
      else if (!strcmp(*argv, "-true_depth")) { argc--; argv++; true_depth_filename = *argv; }
      else if (!strcmp(*argv, "-fx")) { argc--; argv++; camera_intrinsics[0][0] = atof(*argv); }
      else if (!strcmp(*argv, "-fy")) { argc--; argv++; camera_intrinsics[1][1] = atof(*argv); }
      else if (!strcmp(*argv, "-cx")) { argc--; argv++; camera_intrinsics[0][2] = atof(*argv); }
      else if (!strcmp(*argv, "-cy")) { argc--; argv++; camera_intrinsics[1][2] = atof(*argv); }
      else if (!strcmp(*argv, "-xres")) { argc--; argv++; xres = atoi(*argv); }
      else if (!strcmp(*argv, "-yres")) { argc--; argv++; yres = atoi(*argv); }
      else if (!strcmp(*argv, "-inertia_weight")) { argc--; argv++; inertia_weight = atof(*argv); }
      else if (!strcmp(*argv, "-duv_weight")) { argc--; argv++; derivative_weight = atof(*argv); }
      else if (!strcmp(*argv, "-normal_weight")) { argc--; argv++; normal_weight = atof(*argv); }
      else if (!strcmp(*argv, "-tangent_weight")) { argc--; argv++; tangent_weight = atof(*argv); }
      else if (!strcmp(*argv, "-smoothness_weight")) { argc--; argv++; smoothness_weight = atof(*argv); }
      else if (!strcmp(*argv, "-range_weight")) { argc--; argv++; range_weight = atof(*argv); }
      else if (!strcmp(*argv, "-minimum_depth")) { argc--; argv++; minimum_depth = atof(*argv); }
      else if (!strcmp(*argv, "-maximum_depth")) { argc--; argv++; maximum_depth = atof(*argv); }
      else if (!strcmp(*argv, "-png_depth_scale")) { argc--; argv++; png_depth_scale =  atof(*argv); }
      else if (!strcmp(*argv, "-normalize_tangent_vectors")) normalize_tangent_vectors = TRUE;
      else if (!strcmp(*argv, "-gravity")) {
        argc--; argv++; gravity_vector_in_camera_coordinates[0] = atof(*argv);
        argc--; argv++; gravity_vector_in_camera_coordinates[1] = atof(*argv);
        argc--; argv++; gravity_vector_in_camera_coordinates[2] = atof(*argv);
      }
      else {
        fprintf(stderr, "Invalid program argument: %s", *argv);
        exit(1);
      }
    }
    else {
      if (!input_depth_filename) input_depth_filename = *argv;
      else if (!output_depth_filename) output_depth_filename = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    argv++; argc--;
  }

  // Check program arguments
  if (!input_depth_filename || !output_depth_filename) {
    printf("Usage: depth2depth inputdepth outputdepth [options]\n");
    return 0;
  }
  
  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int 
main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read inputs
  if (!ReadInputs()) exit(-1);

  // Create images
  if (!CreateDepthImage()) exit(-1);

  // Write outputs
  if (!WriteOutputs()) exit(-1);

  // Return success
  return 0;
}
