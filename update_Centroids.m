function new_centroids = update_Centroids(data, K)
    % This function computes new centroids using the mean of assigned vectors.
    new_centroids = zeros(K, size(data, 2));
    for i = 1:K
        cluster_indices = find(data(:, 785) == i);
        if ~isempty(cluster_indices)
            new_centroids(i, 1:784) = mean(data(cluster_indices, 1:784), 1);
        else
            % Handle empty clusters by re-initializing the centroid.
            random_idx = randi(size(data, 1));
            new_centroids(i, :) = data(random_idx, :);
        end
    end
end