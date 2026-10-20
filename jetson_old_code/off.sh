#!/bin/bash
# Dung khan cap: pkill ca patrol_robot.py va real_car_socket.py.
# Chay truc tiep tren Jetson (khong can qua SSH/mang) de dam bao
# dung duoc ngay ca khi mang laptop<->Jetson bi rot.
pkill -INT -f patrol_robot.py
pkill -INT -f real_car_socket.py
sleep 1
echo '--- Con lai (neu trong la da dung sach): ---'
ps aux | grep -E 'patrol_robot|real_car_socket' | grep -v grep
