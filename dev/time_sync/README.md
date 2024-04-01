# PTP

* **ptp_server_unicast.conf**: PTP server configuration
* **Validation Guide - Precision Time Protocol.docx**: Microsoft guide on setting PTP client on Windows

## Configuration

On Linux, you should install the PTP application `ptpd` by, e.g., apt or any other manager you like:

```
sudo apt update
sudo apt install ptpd
```

On Windows, you should follow the guidance doc *Validation Guide - Preceision Time Protocol* to set up PTP client. Note that Windows PTP service is only supported on Windows Server 2019 and Windows 10 (v1809) according to Microsoft, and we have no knowledge if the guidance will work or not on the other distributions. Moreoer, Windows PTP can only be used as a client to listen the timestamp from another master machine.

## Synchronize Time

On Linux, you have to explicitly run the PTP program as a master to distribute the timestamps.

***<u>TODO: refer to the Jianing's doc.</u>***

On Windows, there is nothing else you need to do after proper configuration. To test the result of synchronization, you can run TODO: cmd to check offset.