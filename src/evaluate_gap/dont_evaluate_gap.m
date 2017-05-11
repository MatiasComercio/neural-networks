function [eta, alpha, global_error, good_gap] = ...
        dont_evaluate_gap(~, ~, curr_global_error, ...
        ~, eta, alpha, ~)
  % Update global error despite not comparing it with the previous one
  global_error = curr_global_error;
  good_gap = true;
end
