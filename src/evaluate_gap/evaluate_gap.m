function [eta, alpha, global_error, good_gap] = ...
        evaluate_gap(gap, epoch_i, curr_global_error, ...
        prev_global_error, eta, alpha, original_alpha)
  % Constant definitions
  a = 0.001;
  b = 0.1;

  % Variables assignment
  global_error = prev_global_error;
  good_gap = true;
  % It's still no time to make the evaluation
  if mod(epoch_i, gap) ~= 0
      return;
  end
  % Time to make the evaluation
  global_error = curr_global_error;
  % The following threshold is because of the problem of number
  % representation on computers
  if curr_global_error < prev_global_error
    % Improvement was reach during this gap => restore alpha and let eta be
    % more flexible
    eta = eta + a;
    alpha = original_alpha;
    return;
  end
  % Error has been degraded, so, it hadn't been a good gap
  good_gap = false;
  eta = eta - b * eta;
  alpha = 0;
end
