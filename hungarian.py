import numpy as np

def hungarian(perf):
    # Initialize Variables
    matching = np.zeros(perf.shape)

    # Condense the Performance Matrix by removing any unconnected vertices to
    #  increase the speed of the algorithm

    # Find the number in each column that are connected
    num_y = sum(~np.isinf(perf), 1)
    # Find the number in each row that are connected
    num_x = sum(~np.isinf(perf), 2)
    
    # Find the columns(vertices) and rows(vertices) that are isolated
    x_con = np.nonzero(num_x!=0)
    y_con = np.nonzero(num_y!=0)
    
    # Assemble Condensed Performance Matrix
    P_size = max(x_con[0].shape[0], y_con[0].shape[0])
    P_cond = np.zeros((P_size, P_size))
    # P_cond[0:x_con[0].shape[0]-1, 0:y_con[0].shape[0]-1] = perf[x_con[0], y_con[0]]
    P_cond[:, :] = perf[:, :]
    if P_cond.size == 0:
      cost = 0
      return matching, cost

    # Ensure that a perfect matching exists
    # Calculate a form of the Edge Matrix
    Edge = P_cond
    Edge[~np.isinf(P_cond)] = 0
    # Find the deficiency(CNUM) in the Edge Matrix
    cnum = min_line_cover(Edge)

    # Project additional vertices and edges so that a perfect matching
    # exists
    Pmax = max(P_cond[~np.isinf(P_cond)])
    P_size = P_cond.shape[1] + cnum
    P_cond = np.ones((int(P_size[0]), int(P_size[0]))) * Pmax
    #P_cond[1:x_con.shape[1], 1:y_con.shape[1]] = perf[x_con, y_con]
    P_cond = perf
   
    #*************************************************
    # MAIN PROGRAM: CONTROLS WHICH STEP IS EXECUTED
    #*************************************************
    exit_flag = 1
    stepnum = 1
    while exit_flag:
        if stepnum == 1:
            P_cond, stepnum = step1(P_cond)
        elif stepnum == 2:
            r_cov, c_cov, M, stepnum = step2(P_cond)
        elif stepnum == 3:
            c_cov, stepnum = step3(M, P_size)
        elif stepnum == 4:
            M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(P_cond, r_cov, c_cov, M)
        elif stepnum == 5:
            M, r_cov, c_cov, stepnum = step5(M,Z_r,Z_c,r_cov,c_cov)
        elif stepnum == 6:
            P_cond, stepnum = step6(P_cond, r_cov, c_cov)
        elif stepnum == 7:
            exit_flag = 0

    # Remove all the virtual satellites and targets and uncondense the
    # Matching to the size of the original performance matrix.
    #matching[x_con, y_con] = M[1:x_con.shape[1], 1:y_con.shape[1]]
    matching = M
    cost = np.sum(np.ma.masked_where(matching == 0, perf))
    return matching, cost

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   STEP 1: Find the smallest number of zeros in each row
#           and subtract that minimum from its row
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def step1(P_cond):
    P_size = P_cond.shape[1]
    
    # Loop throught each row
    for ii in range(0, P_size):
        rmin = min(P_cond[ii, :])
        P_cond[ii,:] = P_cond[ii, :] - rmin

    stepnum = 2

    return P_cond, stepnum
  
#**************************************************************************  
#   STEP 2: Find a zero in P_cond. If there are no starred zeros in its
#           column or row start the zero. Repeat for each zero
#**************************************************************************

def step2(P_cond):
    # Define variables
    P_size = P_cond.shape[1]
    r_cov = np.zeros((P_size, 1))   # A vector that shows if a row is covered
    c_cov = np.zeros((P_size, 1))   # A vector that shows if a column is covered
    M = np.zeros((P_size, P_size))  # A mask that shows if a position is starred or primed
    
    for ii in range(0, P_size):
        for jj in range(0, P_size):
            if P_cond[ii,jj] == 0 and r_cov[ii] == 0 and c_cov[jj] == 0:
                M[ii,jj] = 1
                r_cov[ii] = 1
                c_cov[jj] = 1
  
    # Re-initialize the cover vectors
    r_cov = np.zeros((P_size, 1)) # A vector that shows if a row is covered
    c_cov = np.zeros((P_size, 1)) # A vector that shows if a column is covered
    stepnum = 3

    return r_cov, c_cov, M, stepnum
  
#**************************************************************************
#   STEP 3: Cover each column with a starred zero. If all the columns are
#           covered then the matching is maximum
#**************************************************************************

def step3(M, P_size):
    c_cov = sum(M, 0)
    if sum(c_cov) == P_size:
        stepnum = 7
    else:
        stepnum = 4

    return c_cov, stepnum
  
#**************************************************************************
#   STEP 4: Find a noncovered zero and prime it.  If there is no starred
#           zero in the row containing this primed zero, Go to Step 5.  
#           Otherwise, cover this row and uncover the column containing 
#           the starred zero. Continue in this manner until there are no 
#           uncovered zeros left. Save the smallest uncovered value and 
#           Go to Step 6.
#**************************************************************************

def step4(P_cond, r_cov, c_cov, M):
    P_size = P_cond.shape[1]

    zflag = 1
    while zflag:
        # Find the first uncovered zero
        row = 0
        col = 0
        exit_flag = 1
        ii = 0
        jj = 0

        while exit_flag:
            if P_cond[ii,jj] == 0 and r_cov[ii] == 0 and c_cov[jj] == 0:
                row = ii
                col = jj
                exit_flag = 0
            jj = jj + 1 
            if jj > P_size-1:
                jj = 0
                ii = ii+1
            if ii > P_size-1:
                exit_flag = 0

        # If there are no uncovered zeros go to step 6
        if row == 0:
            stepnum = 6
            zflag = 0
            Z_r = 0
            Z_c = 0
        else:
            # Prime the uncovered zero
            M[row, col] = 2
            # If there is a starred zero in that row
            # Cover the row and uncover the column containing the zero
            if sum(np.nonzero(M[row, :] == 1)) != 0:
                r_cov[row] = 1
                zcol = np.nonzero(M[row, :] == 1)
                c_cov[zcol] = 0
            else:
                stepnum = 5
                zflag = 0
                Z_r = row
                Z_c = col

    return M, r_cov, c_cov, Z_r, Z_c, stepnum
  
#**************************************************************************
# STEP 5: Construct a series of alternating primed and starred zeros as
#         follows.  Let Z0 represent the uncovered primed zero found in Step 4.
#         Let Z1 denote the starred zero in the column of Z0 (if any). 
#         Let Z2 denote the primed zero in the row of Z1 (there will always
#         be one).  Continue until the series terminates at a primed zero
#         that has no starred zero in its column.  Unstar each starred 
#         zero of the series, star each primed zero of the series, erase 
#         all primes and uncover every line in the matrix.  Return to Step 3.
#**************************************************************************

def step5(M,Z_r,Z_c,r_cov,c_cov):
    zflag = 1
    ii = 1
    while zflag:
        # Find the index number of the starred zero in the column
        rindex = np.nonzero(M[:, Z_c[ii]] == 1)
        if rindex > 0:
            # Save the starred zero
            ii = ii + 1
            # Save the row of the starred zero
            Z_r[ii, 1] = rindex
            # The column of the starred zero is the same as the column of the 
            # primed zero
            Z_c[ii, 1] = Z_c(ii-1)
        else:
            zflag = 0

        # Continue if there is a starred zero in the column of the primed zero
        if zflag == 1:
            # Find the column of the primed zero in the last starred zeros row
            cindex = np.nonzero(M[Z_r[ii], :] == 2)
            ii = ii+1
            Z_r[ii, 1] = Z_r(ii-1)
            Z_c[ii, 1] = cindex;   

    # UNSTAR all the starred zeros in the path and STAR all primed zeros
    for ii in range(1, Z_r.shape[1]):
        if M(Z_r(ii),Z_c(ii)) == 1:
            M[Z_r[ii], Z_c[ii]] = 0
        else:
            M[Z_r[ii], Z_c[ii]] = 1

    # Clear the covers
    r_cov = r_cov * 0
    c_cov = c_cov * 0

    # Remove all the primes
    M[M == 2] = 0

    stepnum = 3

    return M, r_cov, c_cov, stepnum

#**************************************************************************
# STEP 6: Add the minimum uncovered value to every element of each covered
#         row, and subtract it from every element of each uncovered column.  
#         Return to Step 4 without altering any stars, primes, or covered lines.
#**************************************************************************

def step6(P_cond, r_cov, c_cov):
    a = np.nonzero(r_cov == 0)
    b = np.nonzero(c_cov == 0)
    minval = np.min(P_cond[a,b])

    P_cond[np.nonzero(r_cov == 1), :] = P_cond[np.nonzero(r_cov == 1), :] + minval
    P_cond[:, np.nonzero(c_cov == 0)] = P_cond[:, np.nonzero(c_cov == 0)] - minval

    stepnum = 4

    return P_cond, stepnum

def min_line_cover(Edge):
    # Step 2
    r_cov, c_cov, M, stepnum = step2(Edge)
    # Step 3
    c_cov, stepnum = step3(M, Edge.shape[0])
    # Step 4
    M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(Edge, r_cov, c_cov, M)
    # Calculate the deficiency
    cnum = Edge.shape[1] - sum(r_cov) - sum(c_cov)

    return cnum