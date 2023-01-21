clc; clear; close all

% Data Import

IMU0x2Dalpha = importdata("C:\Users\stanl\OneDrive - Arizona State University\College\Rocketry\IMU calibration\tpm_icra2014\icra2014\srcs\IMU0x2Dalpha.mat");
IMU0x2Domega = importdata("C:\Users\stanl\OneDrive - Arizona State University\College\Rocketry\IMU calibration\tpm_icra2014\icra2014\srcs\IMU0x2Domega.mat");

time = IMU0x2Domega(:,1)';

T_init = 50;
t_w = 2;
intervals = zeros(1, length(time));
intervals(1) = ((time(2) - time(1))/2) + time(1)/2;
intervals(2:length(intervals)) = time(2:length(time))-time(1:length(time)-1);

offset_acc_x = 33123;
offset_acc_y = 33276;
offset_acc_z = 32360;
offset_gyro_x = 32768;
offset_gyro_y = 32466;
offset_gyro_z = 32485;

total_sample = length(IMU0x2Domega(:,1));

omega_s = [IMU0x2Domega(:,2)';IMU0x2Domega(:,3)';IMU0x2Domega(:,4)'];
alpha_s = [IMU0x2Dalpha(:,2)';IMU0x2Dalpha(:,3)';IMU0x2Dalpha(:,4)'];

asdf = find(time <= 50, 1, 'last' );
offset_gyro = [mean(IMU0x2Domega(1:asdf,2));mean(IMU0x2Domega(1:asdf,3));mean(IMU0x2Domega(1:asdf,4))];
offset_acc = [mean(IMU0x2Dalpha(1:asdf,2));mean(IMU0x2Dalpha(1:asdf,3));mean(IMU0x2Dalpha(1:asdf,4))];

omega_s_biasfree = omega_s - offset_gyro;



% % 
figure
plot(IMU0x2Dalpha(:,2),'r')
hold on
plot(IMU0x2Dalpha(:,3),'g')
plot(IMU0x2Dalpha(:,4),'b')
title("Raw accelerometer data")
ylabel("Raw Acceleration"); xlabel("Time, [s/100]")
legend("x-axis signal","y-axis signal","z-axis signal")

figure
plot(IMU0x2Domega(:,2),'r')
hold on
plot(IMU0x2Domega(:,3),'g')
plot(IMU0x2Domega(:,4),'b')
title("Raw gyroscope data")
ylabel("Raw Angular Velocity"),xlabel("Time, [s/100]")
legend("x-axis signal","y-axis signal","z-axis signal")

figure
plot(omega_s_biasfree(1,:),'r')
hold on
plot(omega_s_biasfree(2,:),'g')
plot(omega_s_biasfree(3,:),'b')
title("Bias free gyroscope data")
ylabel("Bias Free Angular Velocity"),xlabel("Time, [s/100]")
legend("x-axis signal","y-axis signal","z-axis signal")