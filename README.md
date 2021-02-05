# V1_VioDet
V1版本暴力检测，仅包含了暴力检测算法，尚未包含姿态估计！
使用方法：


可以选择不带参数方式运行， 这时会使用默认的摄像头地址、推流IP、推流窗口编号、socket_address
默认的摄像头地址为  "rtsp://admin:zhks2020@192.168.1.6:554/Streaming/Channels/1"
默认的推流ip为  "192.168.1.201:1936"
默认的推流窗口编号为 2
默认socket_address为 ：192.168.1.199:6666

     python cam_detect.py

也可以选择带参数运行方式， 可以指定摄像头地址、推流IP、推流窗口编号、socket_address
    python cam_detect.py -source_url [SOURCE_URL] -rtmp_address [RTMP_ADDRESS] -cam_index [CAM_INDEX] -socket_address [SOCK_ADDRESS]

# for example ：

## 在有利新配的3*t4服务器上
    python cam_detect.py \
    -source_url "rtsp://admin:zhks2020@192.168.1.6:8000/Streaming/Channels/1" \
    -rtmp_address "192.168.23.104:8083" \
    -cam_index 2 \
    -socket_address "192.168.1.199:6666"


## 23服务器上
     python cam_detect.py \
    -source_url "rtsp://admin:scut123456@192.168.1.64:554/Streaming/Channels/1" \
    -rtmp_address "192.168.1.23:1936" \
    -cam_index 2 \
    -socket_address "192.168.1.199:6666"

