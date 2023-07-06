# NatNet_SDK_client

NatNet客户端，用于在局域网内与NatNet服务器（Motive on Windows）通信
- [ros_node_verion](https://git.code.oa.com/robotics_x_vision/Optitrack_NatNet_client/tree/ros_node_version)
- [jamoca port](https://git.code.oa.com/robotics_x_vision/Optitrack_NatNet_client/tree/jamoca)
- [vision_localization](https://git.code.oa.com/robotics_x_vision/Optitrack_NatNet_client/tree/vision_localization)
  - 作为评估视觉vio的工具，支持将轨迹保存为TUM格式的文件
  - 作为评估视觉vio的工具，支持ros可视化动捕的轨迹结果

<img src="./readme_data/natnet_modified.png" height="400px">

## Dependency

- cmake >= 3.5
- Eigen

## Build

```bash
git clone http://git.code.oa.com/robotics_x_vision/Optitrack_NatNet_client.git
cd natnet_client
mkdir build
cd build
cmake ..
make
```

## Usage

###### windows: optitrack Motive 

###### ubuntu : client

(windows) 启动 Motive软件，载入标定文件，检查刚体跟踪情况

(windows) 显示data Streaming面板 View->dataStreaming

<img src="https://v22.wiki.optitrack.com/images/1/11/DataStreaming_Pane_21.png" height="500px" style="zoom: 80%;" >

(windows) 切换Local Interface至ubuntu和主机所在局域网，打开Boardcast Frame Data选项

(ubuntu)  设置与主机的网络连接，设置新连接，若为以太网直连，为ubuntu设置与主机网段相同的静态IP，若为无线路由器局域网内连接，于WIFI设备手动或自动为win与ubuntu配置网络。`ifconfig` 检查当前IP地址是否和主机同网段，关闭除当前传输网段网络外的其他网络连接。

(ubuntu) 启动示例客户端

```shell
cd ./natnet_client/build/
SERVER_IP="xxx.xxx.xxx.xxx" ./natnet_client
```

## Structure

封装示例:

[example_main.cpp](src/example_main.cpp)

- optitack采用每帧定位数据触发回调函数的机制，以新开辟一个线程不断执行回调函数，故回调函数不应占用过长时间

- 封装类`Optitrack`的回调函数先对数据进行预处理，然后执行用户定义函数[optitrack_collect.cpp](src/optitrack_collect.cpp)=>`Optitrack::ProcessData()`

- ```c++
    /**
     * callback func within Datahandler
     *
     * You can process data here like publish/store...
     * if processing time will be much long, try to use another thread
     * store data in a class member queue (store rigid body data with certain ID reconmended )
     * for example:
     *      boost::lockfree::spsc_queue<MocapData, boost::lockfree::capacity<10> > queue;
     *      queue.push(data);
     *
     * @param data available mocap data
     */
    void ProcessData(const MocapData& data) {
      printf("Rigid Body [ID=%d  Error=%3.4f  frame=%ld]\n", data.id, data.mean_error, data.frame);
    }
  ```

- ```c++
  struct MocapData {
    int id;             		 //rigid_body ID
    long frame;          	 	 //frame num
    double timestamp;     	 //timestamp of curr machine
    double latency_sec;   	 //latency_sec latency from camera capture to receive data
    double mean_error;         //err of localization
    Eigen::Matrix4f T;         //local pose reference to start pose frame
    Eigen::Matrix4f Global_T;  //global pose reference to Optitrack World frame(Y-UP)
  }; 
  
  ```

如需求更多功能，可参考

原始示例：

[SampleClient.cpp](samples/SampleClient/SampleClient.cpp)




## Reference

[NatNet SDK](http://wiki.optitrack.com.cn/index.php/NatNet_SDK)