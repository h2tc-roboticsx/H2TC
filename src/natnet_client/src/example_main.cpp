// Copyright (c) Tencent Corporation. All rights reserved.

#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <arpa/inet.h>
#include <netinet/in.h>
#include "client/optitrack_collect.h"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#define PORT 3003

namespace {
volatile std::sig_atomic_t g_signal;
}  // namespace

/*
  write a matrix as a line into the given file

  @param f: the output file stream
  @param matrix: the matrix to be written

 */

void writeMatrix(std::ofstream &f, Eigen::Matrix4f &matrix) {
  int nrows = matrix.rows(); // number of rows in the matrix
  int ncols = matrix.cols(); // number of columns in the matrix

  // iterate every item in the matrix, i: row index, j: column index
  // write items (separted by ' ') in the order of rows
  for (int i=0, j=0; i<nrows && j<ncols;) {
    f << matrix(i, j) << " ";
    j++;
    if (j == ncols) {
      i++;
      j = 0;
    }
  }
}


/*
  listen to the command from the recorder server i.e. src/recorder.py
  parse and output the received OptiTrack data into a CSV file
  
  @param optitrack: the pointer to the optitrack client instance which receives and processes the data 
  
 */

void listenUDPCommand(opti::Optitrack *optitrack) {
  int sockfd;
  char buffer[10];
  // the message (client ID) to be sent to the recorder server to connect
  const char *msg = "optitrack"; 
  struct sockaddr_in servaddr;

  // initialize socket
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  // exit the program if the socket fails to initialize
  if (sockfd < 0){
    perror("socket creation failed");
    exit(EXIT_FAILURE);
  }

  // configurate the socket
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = inet_addr("10.41.206.18");
  unsigned int len; // the length of the received message from the recorder server
  // send client ID to the recorder server to connect
  sendto(sockfd,
	 (const char*)msg,
	 strlen(msg),
	 MSG_CONFIRM,
	 (const struct sockaddr*)&servaddr,
	 sizeof(servaddr));

  // infinite loop to keep waiting for and processing the command from the recorder server
  while (true) {
    printf("Waiting for next command:\n");
    // block to receive the message from the recorder server
    recvfrom(sockfd, (char*)buffer, 10, MSG_WAITALL, (struct sockaddr*) &servaddr, &len);

    // break the loop to exit the program if the message is '-1'
    if (buffer[0]=='-' && buffer[1]=='1') break;
    // continue to the next iteration if the message is '-2' i.e. connection confirmed
    else if (buffer[0]=='-' && buffer[1]=='2') continue;

    // enable the saving of the received data by setting the recording variable of optitrack
    optitrack->recording = true;
    // sleep the length of recording i.e. 5 seconds
    sleep(5);
    // disable the saving of the received data by setting the recording variable of optitrack
    optitrack->recording = false;
    opti::MocapData data;
    std::ofstream f;
    char filename[100];
    // generate the output filepath
    sprintf(filename, "../../../data/%s/optitrack.csv", buffer);
    // open the output file stream as f
    f.open(filename);

    // iterate every frame stored in the queue of optitrack
    // a frame includes in order:
    // object ID, frame number, mean localization error, timestamp, latency,
    // local transformation matrix, global transformation matrix
    // each entry is separated by ' '
    while (optitrack->queue.pop(data)){
      f << data.id << " ";
      f << data.frame << " ";
      f << data.mean_error << " ";
      f << std::fixed << data.timestamp << " " << data.latency_sec << " ";
      writeMatrix(f, data.T);
      writeMatrix(f, data.Global_T);
      f << std::endl;
    }
    
    f.close(); // close the file stream
  }
  close(sockfd); // close the socket
}


int main(int argc, char** argv) {
  char* ip_address = "10.41.206.52"; // OptiTrack server IP

e  // initialize OptiTrack instance to receive data from OptiTrack server
  opti::Optitrack optitrack;
  optitrack.SetIp(ip_address);
  optitrack.Start();

  // start a thread to listen to the message from the recorder server
  void listenUDPCommand(opti::Optitrack *optitrack);
  std::thread t(listenUDPCommand, &optitrack);
  
  t.join(); // wait for the completion of the listening thread
  optitrack.Stop(); // stop receiving the data from OptiTrack server
  return 0;
}
