function [eta, alpha, good_gap] = evaluate_gap(prev_gap_global_error, ...
    curr_gap_global_error, eta, ~, original_alpha)
	% Constant definitions
	a = 0.001;
	b = 0.1;
  
	% Error has been degraded, so, it not a good gap
	if (curr_gap_global_error >= prev_gap_global_error)
        good_gap = false;
        eta = eta - b * eta;
        alpha = 0;
      
        return;
	end

    % Improvement was reach during this gap
    %  => Restore alpha and let eta be more flexible
    good_gap = true;
    eta = eta + a;
    alpha = original_alpha;
end
