function [index, vec_distance] = assign_vector_to_centroid(data_vec, centroids)
    % This function finds the closest centroid for a given data vector.
    num_centroids = size(centroids, 1);
    min_dist = inf; 
    index = -1;
    for i = 1:num_centroids
        dist = norm(data_vec - centroids(i, 1:784));
        if dist < min_dist
            min_dist = dist;
            index = i;
        end
    end
    vec_distance = min_dist;
end