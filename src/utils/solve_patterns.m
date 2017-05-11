function [outputs, memories] = solve_patterns(net, patterns)
% SOLVE_PATTERNS Use the network to solve each test pattern
   for i = 1:columns(patterns)
       [output, memory] = net.solve(net, patterns(:,i));
       outputs(i) = output;
       memories(i).memory = memory;
   end
end
