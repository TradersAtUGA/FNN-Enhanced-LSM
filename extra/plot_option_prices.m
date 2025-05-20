% Read data from output.txt
filename = 'output.txt';
data = fileread(filename);

% Split the data into lines
lines = strsplit(data, '\n');
num_lines = length(lines);

% Initialize arrays to store Option Prices and Polynomial Degrees
option_prices = [];
poly_degrees = [];

% Loop through each line and extract the Option Price and Polynomial Degree
for i = 1:num_lines-1
    % Split the line into parts based on 'Option Price:' and 'Polynomial Degree:'
    parts = strsplit(lines{i});
    
    % Extract the option price (assuming it's after 'Option Price:' part)
    option_price = str2double(parts{3});
    % Extract the polynomial degree (assuming it's after 'Polynomial Degree:' part)
    poly_degree = str2double(parts{6});
    
    % Store the values
    option_prices = [option_prices, option_price];
    poly_degrees = [poly_degrees, poly_degree];
end

% Plot the data
figure;
plot(poly_degrees, option_prices, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Polynomial Degree');
ylabel('Option Price');
title('Option Price vs Polynomial Degree');
grid on;
