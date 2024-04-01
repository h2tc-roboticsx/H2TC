// Copyright (c) Tencent Corporation. All rights reserved.


#include "client/optitrack_collect.h"

#include <unistd.h>
#include <chrono>


namespace opti {
namespace {

auto Now() {
  // return the current UNIX timestamp
  auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()
    );
  return now.count();
}

// DataHandler receives data from the server
// This function is called by NatNet when a frame of mocap data is available
void DataHandler(sFrameOfMocapData *data, void *user_data) {
  auto *optitrack = reinterpret_cast<Optitrack *>(user_data);
  const NatNetClient *client = optitrack->Client();
  //  const sServerDescription *server_description = optitrack->ServerDescription();
  // Transit latency is defined as the span of time between Motive transmitting the frame of data,
  // and its reception by the client (now). The SecondsSinceHostTimestamp method relies on
  // NatNetClient's internal clock synchronization with the server using Cristian's algorithm.
  const double transitLatencyMillisec =
      client->SecondsSinceHostTimestamp(data->TransmitTimestamp) * 1e3;
  const double latencyMillisec =
      client->SecondsSinceHostTimestamp(data->CameraMidExposureTimestamp) * 1e3;

  // the frame timestamp = current timestamp - total latency
  const double curr_timestamp = Now() - latencyMillisec * 1e6;
  debug("total_Latency : %.2lf milliseconds\n", latencyMillisec);
  debug("transmit_Latency : %.2lf milliseconds\n", transitLatencyMillisec);

  debug("FrameID : %d\n", data->iFrame);
  debug("Timestamp : %3.2lf\n", data->fTimestamp);

  // Rigid Bodies
  debug("Rigid Bodies [Count=%d]\n", data->nRigidBodies);

  for (int i = 0; i < data->nRigidBodies; ++i) {
    // params
    // 0x01 : bool, rigid body was successfully tracked in this frame
    bool bTrackingValid = data->RigidBodies[i].params & 0x01;
    if (bTrackingValid) {
      // optitrack format to Eigen::Matrix
      Eigen::Matrix4f curr_global_pose = Eigen::Matrix4f::Identity();
      Eigen::Quaternionf curr_global_quat(data->RigidBodies[i].qw, data->RigidBodies[i].qx,
                                          data->RigidBodies[i].qy, data->RigidBodies[i].qz);
      curr_global_pose(0, 3) = data->RigidBodies[i].x;
      curr_global_pose(1, 3) = data->RigidBodies[i].y;
      curr_global_pose(2, 3) = data->RigidBodies[i].z;
      curr_global_pose.block(0, 0, 3, 3) = curr_global_quat.toRotationMatrix();
      // check init
      std::pair<int, Eigen::Matrix4f> curr_status;
      std::pair<int, Eigen::Matrix4f> * curr_status_ptr = &curr_status;
      if (optitrack->get_body_status(data->RigidBodies[i].ID, curr_status_ptr)) {
        MocapData optitrack_data;
        optitrack_data.id = data->RigidBodies[i].ID;
        optitrack_data.timestamp = curr_timestamp;
        optitrack_data.frame = data->iFrame;
        optitrack_data.latency_sec = latencyMillisec / 1e3;
        optitrack_data.mean_error = data->RigidBodies[i].MeanError;
        optitrack_data.T = curr_status.second.inverse() * curr_global_pose;
        optitrack_data.Global_T = curr_global_pose;
        optitrack->ProcessData(optitrack_data);
        debug("\tID\tframeID\t\tx\ty\tz\tlatency(s)\terr\n");
        debug("\t%d\t%d\t%3.2f\t%3.2f\t%3.2f\t%.4f\t\t%.4f\n", data->RigidBodies[i].ID,
              data->iFrame, optitrack_data.T(0, 3), optitrack_data.T(1, 3), optitrack_data.T(2, 3),
              optitrack_data.latency_sec, data->RigidBodies[i].MeanError);
      } else {
        optitrack->body_status.emplace_back(data->RigidBodies[i].ID, curr_global_pose);
      }
    }
  }
}
}  // namespace

/*
  process the data from OptiTrack server

  @praram data: encapsule of the received data
 */
void Optitrack::ProcessData(const MocapData& data) {
  // do not process if recordign is off
  if (!recording) return;
  bool target = false; // determine if the received data belonging to the target object
  // loop every target ID to check if anyone matches the ID of the received data
  // if so, set the target variable to be true
  for (auto it=ids.begin(); it!=ids.end(); it++) {if (*it == data.id) target=true;}
  // skip data not belonging to the target object
  if (!target) return;
  printf("Rigid Body [ID=%d  frame=%ld  time=%3.2lf]\n", data.id, data.frame, data.timestamp);
  // store the data in the queue to be processed later in the main process
  queue.push(data);
}
  
bool Optitrack::get_body_status(int id, std::pair<int, Eigen::Matrix4f> *rb_status) {
  if (!body_status.empty()) {
    for (const auto &s : body_status) {
      if (s.first == id) {
        *rb_status = s;
        return true;
      }
    }
  }
  return false;
}

void Optitrack::Start() {
  // print version info
  unsigned char ver[4];
  NatNet_GetVersion(ver);
  debug("NatNet Sample Client (NatNet ver. %d.%d.%d.%d)\n", ver[0], ver[1], ver[2], ver[3]);

  // create NatNet client
  client_ = std::make_unique<NatNetClient>();
  // set the frame callback handler
  client_->SetFrameReceivedCallback(DataHandler, this);

  printf("Mocap IP: %s\n", server_ip_.c_str());
  connect_params.serverAddress = server_ip_.c_str();
  
  int rv;

  // Connect to Motive
  rv = ConnectClient();
  if (rv != ErrorCode_OK) {
    printf("Error initializing client.  See log for details.  Exiting\n");
    return;
  } else {
    printf("Client initialized and ready.\n");
  }
  // Ready to receive marker stream!
  printf("\nClient is connected to server and listening for data...\n");

}

  
// Establish a NatNet Client connection
int Optitrack::ConnectClient() {
  // Release previous server
  client_->Disconnect();

  // Init Client and connect to NatNet server
  int retCode = client_->Connect(connect_params);
  if (retCode != ErrorCode_OK) {
    debug("Unable to connect to server.  Error code: %d. Exiting", retCode);
    return ErrorCode_Internal;
  } else {
    // connection succeeded

    void *result;
    int bytes = 0;
    ErrorCode ret = ErrorCode_OK;

    // print server info
    memset(&server_description_, 0, sizeof(server_description_));
    ret = client_->GetServerDescription(&server_description_);
    if (ret != ErrorCode_OK || !server_description_.HostPresent) {
      printf("Unable to connect to server. Host not present. Exiting.");
      return 1;
    }
    debug("\n[SampleClient] Server application info:\n");
    debug("Application: %s (ver. %d.%d.%d.%d)\n", server_description_.szHostApp,
          server_description_.HostAppVersion[0], server_description_.HostAppVersion[1],
          server_description_.HostAppVersion[2], server_description_.HostAppVersion[3]);
    debug("NatNet Version: %d.%d.%d.%d\n", server_description_.NatNetVersion[0],
          server_description_.NatNetVersion[1], server_description_.NatNetVersion[2],
          server_description_.NatNetVersion[3]);
    debug("Client IP:%s\n", connect_params.localAddress);
    debug("Server IP:%s\n", connect_params.serverAddress);
    debug("Server Name:%s\n", server_description_.szHostComputerName);

    // get mocap frame rate
    ret = client_->SendMessageAndWait("FrameRate", &result, &bytes);
    if (ret == ErrorCode_OK) {
      float rate = *(reinterpret_cast<float *>(result));
      debug("Mocap Framerate : %3.2f\n", rate);
    } else {
      printf("Error getting frame rate.\n");
    }

    // get # of analog samples per mocap frame of data
    ret = client_->SendMessageAndWait("AnalogSamplesPerMocapFrame", &result, &bytes);
    if (ret == ErrorCode_OK) {
      int analog_samples_per_frame = *(reinterpret_cast<int *>(result));
      debug("Analog Samples Per Mocap Frame : %d\n", analog_samples_per_frame);
    } else {
      debug("Error getting Analog frame rate.\n");
    }
  }
  return ErrorCode_OK;
}

}  // namespace opti
