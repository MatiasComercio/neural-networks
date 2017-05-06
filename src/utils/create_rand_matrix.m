function matrix = create_rand_matrix(min, max, rows, cols)
% min: inclusive
% max: inclusive
% rows: matrix's number of rows
% cols: matrix's number of columns
  matrix = (max-min).*rand(rows, cols) + min;
end

