static int
CreateNormalEquations(RNSystemOfEquations& equations)
{
  // Create normal equations
  if ((input_nx_image && (nx_weight > 0)) || (input_ny_image && (ny_weight > 0)) ||
      (input_nz_image && (nz_weight > 0)) || (input_wnz_image && (wnz_weight > 0))) {
    // Check camera intrinsics
    if (RNIsZero(camera_intrinsics[0][0]) || RNIsZero(camera_intrinsics[1][1])) {  
      fprintf(stderr, "You must provide camera intrinsics to create normal equations\n");
      return 0;
    }

    // Create normal equations
    for (int iy = 0; iy < yres; iy++) {
      for (int ix = 0; ix < xres; ix++) {
        RNScalar input_nx = (input_nx_image) ? input_nx_image->GridValue(ix, iy) : 0;
        if (input_nx == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar input_ny = (input_ny_image) ? input_ny_image->GridValue(ix, iy) : 0;
        if (input_ny == R2_GRID_UNKNOWN_VALUE) continue;
        RNScalar input_nz = (input_nz_image) ? input_nz_image->GridValue(ix, iy) : 0;
        if (input_nz == R2_GRID_UNKNOWN_VALUE) continue;

        // Consider triangles in 1-4 directions
        for (int dir = 0; dir < 1; dir++) {
          // Get image coordinates of two adjacent points
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

          // Compute vectors between points
          RNPolynomial dxA = xA - x;
          RNPolynomial dxB = xB - x;
          RNPolynomial dyA = yA - y;
          RNPolynomial dyB = yB - y;
          RNPolynomial dzA = d - dA;
          RNPolynomial dzB = d - dB;

          // Compute cross product of vectors
          RNAlgebraic *cx1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNPolynomial(dyA), new RNPolynomial(dzB));
          RNAlgebraic *cx2 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNPolynomial(dzA), new RNPolynomial(dyB));
          RNAlgebraic *cx = new RNAlgebraic(RN_SUBTRACT_OPERATION, cx1, cx2);
          RNAlgebraic *cy1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNPolynomial(dzA), new RNPolynomial(dxB));
          RNAlgebraic *cy2 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNPolynomial(dxA), new RNPolynomial(dzB));
          RNAlgebraic *cy = new RNAlgebraic(RN_SUBTRACT_OPERATION, cy1, cy2);
          RNAlgebraic *cz1 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNPolynomial(dxA), new RNPolynomial(dyB));
          RNAlgebraic *cz2 = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNPolynomial(dyA), new RNPolynomial(dxB));
          RNAlgebraic *cz = new RNAlgebraic(RN_SUBTRACT_OPERATION, cz1, cz2);

          // Compute length of cross product
          RNAlgebraic *xslen = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*cx), 2);
          RNAlgebraic *yslen = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*cy), 2);
          RNAlgebraic *zslen = new RNAlgebraic(RN_POW_OPERATION, new RNAlgebraic(*cz), 2);
          RNAlgebraic *slen = new RNAlgebraic(RN_ADD_OPERATION, xslen, yslen);
          slen = new RNAlgebraic(RN_ADD_OPERATION, slen, zslen);
          RNAlgebraic *len = new RNAlgebraic(RN_POW_OPERATION, slen, 0.5);

          // Add equations for error in world normal
          if (input_wnz_image && (wnz_weight > 0)) {
            RNScalar input_wnz = input_wnz_image->GridValue(ix, iy);
            if (input_wnz != R2_GRID_UNKNOWN_VALUE) {
              RNAlgebraic *wnz = new RNAlgebraic(RN_MULTIPLY_OPERATION, input_wnz, new RNAlgebraic(*len));
              RNAlgebraic *dotx = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNAlgebraic(*cx), gravity_vector_in_camera_coordinates[0]);
              RNAlgebraic *doty = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNAlgebraic(*cy), gravity_vector_in_camera_coordinates[1]);
              RNAlgebraic *dotz = new RNAlgebraic(RN_MULTIPLY_OPERATION, new RNAlgebraic(*cz), gravity_vector_in_camera_coordinates[2]);
              RNAlgebraic *dot = new RNAlgebraic(RN_ADD_OPERATION, dotx, doty);
              dot = new RNAlgebraic(RN_ADD_OPERATION, dot, dotz);
              RNAlgebraic *e = new RNAlgebraic(RN_SUBTRACT_OPERATION, dot, wnz);
              e = new RNAlgebraic(RN_MULTIPLY_OPERATION, e, wnz_weight);
              equations.InsertEquation(e);
            }
          }

          // Add equations for error in camera normals
          if (input_nx_image && (nx_weight > 0)) {
            RNAlgebraic *nx = new RNAlgebraic(RN_MULTIPLY_OPERATION, input_nx, new RNAlgebraic(*len));
            RNAlgebraic *ex = new RNAlgebraic(RN_SUBTRACT_OPERATION, cx, nx);
            ex = new RNAlgebraic(RN_MULTIPLY_OPERATION, ex, nx_weight);
            equations.InsertEquation(ex);
          }
          if (input_ny_image && (ny_weight > 0)) {
            RNAlgebraic *ny = new RNAlgebraic(RN_MULTIPLY_OPERATION, input_ny, new RNAlgebraic(*len));
            RNAlgebraic *ey = new RNAlgebraic(RN_SUBTRACT_OPERATION, cy, ny);
            ey = new RNAlgebraic(RN_MULTIPLY_OPERATION, ey, ny_weight);
            equations.InsertEquation(ey);
          }
          if (input_nz_image && (nz_weight > 0)) {
            RNAlgebraic *nz = new RNAlgebraic(RN_MULTIPLY_OPERATION, input_nz, new RNAlgebraic(*len));
            RNAlgebraic *ez = new RNAlgebraic(RN_SUBTRACT_OPERATION, cz, nz);
            ez = new RNAlgebraic(RN_MULTIPLY_OPERATION, ez, nz_weight);
            equations.InsertEquation(ez);
          }

          // Delete temporary data
          if (nx_weight == 0) delete cx;
          if (ny_weight == 0) delete cy;
          if (nz_weight == 0) delete cz;
          delete len;
        }
      }
    }
  }

  // Return success
  return 1;
}





