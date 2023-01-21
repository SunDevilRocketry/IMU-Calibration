clc; clear; close all

%% Data Import
IMU0x2Dalpha = importdata("IMU0x2Dalpha.mat");
IMU0x2Domega = importdata("IMU0x2Domega.mat");

%% Initial Condition
time = IMU0x2Domega(:,1)';
total_sample = length(time);
T_init = 50;                        % Initial static period
t_wait = [2,4];                     % Static intervals
k = 37;                             % Number of rotations
offset_acc = [33123;33276;32360];   % Offsets taken from manufact
offset_gyro = [32768;32466;32485];

alpha_s = [IMU0x2Dalpha(:,2)'-offset_acc(1);
    IMU0x2Dalpha(:,3)'-offset_acc(2);
    IMU0x2Dalpha(:,4)'-offset_acc(3)];
omega_s = [IMU0x2Domega(:,2)'-offset_gyro(1);
    IMU0x2Domega(:,3)'-offset_gyro(2);
    IMU0x2Domega(:,4)'-offset_gyro(3)];

T_init_index = find(time <= T_init, 1, 'last' );
gyro_bias = [mean(omega_s(1,1:T_init_index));
    mean(omega_s(2,1:T_init_index));
    mean(omega_s(3,1:T_init_index))];

omega_biasfree = omega_s - gyro_bias;

M_inf = [];

% Variance based static detector operatior
T_init_index = 3000;
var_3D = (var(alpha_s(1,1:T_init_index)))^2+ ...
    (var(alpha_s(2,1:T_init_index)))^2+ ...
    var((alpha_s(3,1:T_init_index)))^2;

t_w = 101%find(time <= 1, 1, 'last');   % Time interval length
half_t_w = floor(t_w/2);

for i = half_t_w+1:total_sample-(half_t_w+1)
    a_t(:,i) = [var(alpha_s(1,i-half_t_w:i+half_t_w));
        var(alpha_s(2,i-half_t_w:i+half_t_w));
        var(alpha_s(3,i-half_t_w:i+half_t_w))];
end

s_square = a_t(1,:).^2+a_t(2,:).^2+a_t(3,:).^2;

s_filter = zeros(1,total_sample);

for i = 1:10
    for n = half_t_w:total_sample - (half_t_w + 1)
    
        if s_square(n) < i*var_3D
        
            s_filter(n) = 1;
        
        end
    
    end
end

% for i = 1:k
%     threshold = i*sigma_init^2;
% 
% end

%% Plots
figure
plot(alpha_s(1,:),'r')
hold on
plot(alpha_s(2,:),'g')
plot(alpha_s(3,:),'b')
ylabel("Raw acceleration");xlabel("Time, [s/100]")
legend("x-axis","y-axis","z-axis")
title("Raw Accelerometer Data")

figure
plot(omega_s(1,:),'r')
hold on
plot(omega_s(2,:),'g')
plot(omega_s(3,:),'b')
ylabel("Raw Angular Velocity");xlabel("Time, [s/100]")
legend("x-axis","y-axis","z-axis")
title("Raw Gyroscope Data")