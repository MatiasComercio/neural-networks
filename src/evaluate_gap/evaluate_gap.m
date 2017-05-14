function [eta, alpha, good_gap] = evaluate_gap(curr_error_variation, eta, ~, original_alpha, error_variation)
        config = get_config('evaluate_gap');
        a = config.a;
        b = config.b;
        k = config.k;

        % Error has been degraded, so, it not a good gap
        if (curr_error_variation > 0)
            good_gap = false;
            eta = eta - b * eta;
            alpha = 0;      
            return;
        end
        
        % Increase eta only if error variation decreased consistently in
        % the previous k epochs
        if (curr_error_variation < 0 && ...
            error_consistently_decreased(error_variation, k))
                eta = eta + a;
        end

        % Improvement was reach during this gap
        %  => Restore alpha and let eta be more flexible
        good_gap = true;
        alpha = original_alpha;
end

function [ error_consistently_decreased ] = error_consistently_decreased(error_variation, k)
    if (length(error_variation) < k)
        error_consistently_decreased = false;
        return;
    end
    last_k_errors = error_variation(length(error_variation)-k+1:length(error_variation));
    error_consistently_decreased = true;
    for i=1:k-1
        if (last_k_errors(i+1)>last_k_errors(i+1))
            error_consistently_decreased = false;
            return;
        end
    end
end