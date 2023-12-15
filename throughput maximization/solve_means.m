function results = solve_means(L1, L2)
    results = zeros((L1-1)*(L1-2)*(L2-1)/2, 1);
    count = 1;
    for s1 = 1:(L1-2)
        for s2 = 1:(L1-s1-1)
            s3 = L1 - s1 - s2;
            for s4 = 1:(L2-1)
                s5 = L2 - s4;
                input = [s1, s2, s3, s4, s5];
                result = lpsolvefun(input);
                results(count) = result;
                count = count +1;
            end
        end
    end
    writematrix(results,['results_' num2str(L1) '_' num2str(L2) '.csv']);
    % means = results;
    % means = -sort(-means);
    % n_max = sum(means >= max(means) - 0.00001);
    % n_good = sum(means >= max(means) - 0.01);
    % disp([max(means), L1, L2, n_max, n_good, means(n_max) - means(n_max+1)])
end