// Copyright (c) Tencent Corporation. All rights reserved.

#ifndef OPTITRACK_NATNET_CLIENT_INCLUDE_CLIENT_OPTITRACK_COLLECT_H_
#define OPTITRACK_NATNET_CLIENT_INCLUDE_CLIENT_OPTITRACK_COLLECT_H_

#include <Eigen/Dense>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/lockfree/spsc_queue.hpp>

#include "natnet/NatNetCAPI.h"
#include "natnet/NatNetClient.h"
#include "natnet/NatNetTypes.h"

// #define __DEBUG___

#ifdef __DEBUG___
#define debug(...) printf(__VA_ARGS__)
#else
#define debug(...) (__VA_ARGS__)
#endif


using Mat4f = Eigen::Matrix4f;
using Mat3f = Eigen::Matrix3f;
using Vec3f = Eigen::Vector3f;
using Quatf = Eigen::Quaternionf;
namespace opti {

struct MocapData {
  int id;              // rigid_body ID
  u_int64_t frame;     // frame num
  double timestamp;    // timestamp of curr machine
  double latency_sec;  // latency_sec latency from camera capture to receive data
  double mean_error;   // err of localization
  Mat4f T;             // local pose reference to start pose frame
  Mat4f Global_T;      // global pose reference to Optitrack World frame(Y-UP)
};
 
class Optitrack {
 public:
  Optitrack() = default;
  ~Optitrack() {}
  bool recording = false;
  boost::lockfree::spsc_queue<MocapData, boost::lockfree::capacity<10000> > queue;
  std::vector<int> ids{115, 116, 117, 118, 101,102,103,104,105,106,107,108,109,110,111,112,113,114,119};
  
  void SetIp(const std::string& ip) { server_ip_ = ip; }
  /**
   * start connection thread
   */
  void Start();

  /**
   * clear connection
   */
  void Stop() {
    if (client_) {
      printf("Stop client\n");
      client_->Disconnect();
      client_.release();
    }
  }

  /**
   * check if current rigid body is initialised
   * @param[in] id   id  of body
   * @param[out] rb_status  body_status with given id
   * @return  exist
   */
  bool get_body_status(int id, std::pair<int, Eigen::Matrix4f>* rb_status);

  /**
   * callback func within Datahandler
   *
   * You can process data here like publish/store...
   * if processing time will be much long, try to use another thread
   * or store data in a class member queue (store rigid body data with certain ID)
   * for example:
   *      boost::lockfree::spsc_queue<MocapData, boost::lockfree::capacity<10> > queue;
   *      queue.push(data);
   *
   * @param data available mocap data
   */
  void ProcessData(const MocapData& data);
  const sServerDescription* ServerDescription() const { return &server_description_; }
  const NatNetClient* Client() const { return client_.get(); }

  std::vector<std::pair<int, Eigen::Matrix4f>> body_status;  // rigid bodies' status

 private:
  int ConnectClient();

  std::string server_ip_;
  std::unique_ptr<NatNetClient> client_;

  // Natnet parameters
  sNatNetClientConnectParams connect_params;
  sServerDescription server_description_;
};
}  // namespace opti

#endif  // OPTITRACK_NATNET_CLIENT_INCLUDE_CLIENT_OPTITRACK_COLLECT_H_
