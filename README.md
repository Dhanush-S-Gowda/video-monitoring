# FogVision: Edge-Fog Based Intelligent Video Monitoring System

FogVision is an intelligent real-time video monitoring system built using Raspberry Pi, edge AI, and fog computing principles. The system performs object detection at the edge device and offloads computationally intensive tasks such as object tracking and face recognition to a fog server.

The platform provides a live monitoring dashboard where employers or administrators can monitor employee movement, entry/exit timings, and surveillance analytics in real time.

\---

# Features

* Real-time object detection on Raspberry Pi
* Camera-based live monitoring
* Frame transmission from edge device to fog server
* Object tracking on fog server
* Face recognition integration
* Employee entry and exit monitoring
* Live dashboard for surveillance analytics
* Distributed edge-fog architecture
* Scalable monitoring pipeline
* Low-latency video analytics

\---

# System Architecture

```text
+-------------------+
| Raspberry Pi Node |
|-------------------|
| Camera Feed       |
| Object Detection  |
+---------+---------+
          |
          | Detected Objects + Frames
          v
+-------------------+
| Fog Server        |
|-------------------|
| Object Tracking   |
| Face Recognition  |
| Event Processing  |
+---------+---------+
          |
          v
+-------------------+
| Live Dashboard    |
|-------------------|
| Employee Logs     |
| Entry/Exit Time   |
| Monitoring UI     |
| Analytics         |
+-------------------+
```

\---

# Tech Stack

## Edge Device

* Raspberry Pi
* Python
* OpenCV

## AI / Computer Vision

* Object Detection Models
* Face Recognition
* Object Tracking Algorithms

## Backend / Fog Layer

* Python
* Flask

## Dashboard

* HTML
* CSS
* JavaScript

\---

# Workflow

1. Camera connected to Raspberry Pi captures live video feed.
2. Raspberry Pi performs object detection locally.
3. Detected object metadata and frames are sent to the fog server.
4. Fog server performs:

   * Object tracking
   * Face recognition
   * Event processing
5. Results are updated on the live dashboard.
6. Dashboard displays:

   * Employee entry time
   * Exit time
   * Monitoring analytics
   * Live surveillance updates

\---

# Use Cases

* Smart office monitoring
* Employee attendance tracking
* Restricted area monitoring
* Industrial surveillance
* Smart campus security
* Real-time workforce analytics

\---

# Advantages of Edge-Fog Architecture

* Reduced cloud dependency
* Lower latency
* Faster response time
* Efficient bandwidth usage
* Better scalability
* Distributed computation

\---

# Future Improvements

* Multi-camera support
* Cloud synchronization
* AI-based anomaly detection
* Person re-identification
* Mobile application integration
* Alert and notification system
* Heatmap analytics
* Real-time crowd analysis

\---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Dhanush-S-Gowda">
      <img src="https://avatars.githubusercontent.com/Dhanush-S-Gowda" width="50px;" alt="Dhanush S Gowda"/><br />
      <b>Dhanush S Gowda</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/veritaserum77">
      <img src="https://avatars.githubusercontent.com/veritaserum77" width="50px;" alt="veritaserum77"/><br />
      <b>veritaserum77</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/bhuvanlord0602">
      <img src="https://avatars.githubusercontent.com/bhuvanlord0602" width="50px;" alt="bhuvanlord0602"/><br />
      <b>bhuvanlord0602</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/JAIPREET-18">
      <img src="https://avatars.githubusercontent.com/JAIPREET-18" width="50px;" alt="JAIPREET-18"/><br />
      <b>JAIPREET-18</b>
      </a>
    </td>
  </tr>
</table>
