%% --- Finding Optimal K (Elbow Method) ---
% NOTE: This is a separate analysis. Run this script first to find
% the best 'k' to use in your main clustering script.

clear;
close all;

train = csvread('mnist_train_1500.csv');
train = train(:,1:784);
train(:,785) = zeros(1500,1);

% --- Define the range of k to test and initialize variables ---
fprintf('Finding Optimal K with the Elbow Method Analysis...\n');
k_values = 10:25; % Range of k to test
costs = zeros(length(k_values), 1);
max_iter_elbow = 25; % Use fewer iterations for this analysis to save time

% --- Main loop to test each k value ---
for i = 1:length(k_values)
    current_k = k_values(i);
    fprintf('Testing k = %d...\n', current_k);
    
    % Run a full k-means clustering for the current_k
    temp_train_data = train;
    temp_centroids = initialize_centroids(temp_train_data, current_k);
    
    for iter = 1:max_iter_elbow
        % Assignment Step
        for j = 1:size(temp_train_data, 1)
            [idx, ~] = assign_vector_to_centroid(temp_train_data(j, 1:784), temp_centroids);
            temp_train_data(j, 785) = idx;
        end
        % Update Step
        temp_centroids = update_Centroids(temp_train_data, current_k);
    end
    
    % Calculate and store the final cost for the current_k
    final_cost = 0;
    for j = 1:size(temp_train_data, 1)
        assigned_centroid_idx = temp_train_data(j, 785);
        if assigned_centroid_idx > 0
            dist = norm(temp_train_data(j, 1:784) - temp_centroids(assigned_centroid_idx, 1:784));
            final_cost = final_cost + dist^2;
        end
    end
    costs(i) = final_cost;
end

%% --- Find the Optimal K Algorithmically ---
first_point = [k_values(1), costs(1)];
last_point = [k_values(end), costs(end)];
line_vec = last_point - first_point;
max_dist = -1;
optimal_k = -1;

for i = 1:length(k_values)
    current_point = [k_values(i), costs(i)];
    point_vec = current_point - first_point;
    distance = abs(point_vec(2)*line_vec(1) - point_vec(1)*line_vec(2)) / norm(line_vec);
    
    if distance > max_dist
        max_dist = distance;
        optimal_k = k_values(i);
    end
end
fprintf('The optimal k value is algorithmically determined to be: %d\n', optimal_k);

%% --- Plot the Elbow Curve ---
figure;
plot(k_values, costs, '-o', 'LineWidth', 2);
hold on;
plot(optimal_k, costs(k_values == optimal_k), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
hold off;
title('Elbow Method for Optimal k');
xlabel('Number of Clusters (k)');
ylabel('Total Cost (Within-Cluster Sum of Squares)');
legend('Cost Curve', sprintf('Optimal k = %d', optimal_k));
grid on;
saveas(gcf, 'elbow_curve.png');