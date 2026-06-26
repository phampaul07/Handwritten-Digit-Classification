function y = initialize_centroids(data, num_centroids)
    % This function randomly chooses k vectors from data to be initial centroids.
    random_index = randperm(size(data,1));
    centroids = data(random_index(1:num_centroids), :);
    y = centroids;
end