# OpenTripPlanner (OTP) Docker Service Guide

This document explains how to create and start a local OpenTripPlanner (OTP) Docker service using the existing `graph.obj` file.

The project currently uses this OTP GraphQL endpoint:

```text
http://localhost:8080/otp/gtfs/v1
```

The project already includes a directly loadable OTP graph and its build data:

```text
data/raw/network/
├─ graph.obj
├─ build-config.json
├─ israel-and-palestine.osm.pbf
├─ jerusalem_gtfs.zip
└─ jerusalem_feed_original.zip
```

For routine use, load the existing `graph.obj` directly. Rebuild the graph only after changing the OSM data, GTFS feed, or graph-build configuration.

---

## 1. Prerequisites

Make sure Docker Desktop is running.

Assume the OTP data directory is:

```text id="vwtkmt"
G:\HUJI-FMCDRT\FMC15_TravelAnalysis_V6\otp_jerusalem\data
```

and that it already contains:

```text id="rte2yq"
graph.obj
```

Verify it with:

```powershell id="ys3o6m"
Get-Item .\data\graph.obj
```

---

## 2. Check or Download the OTP Docker Image

First check whether the OTP image is available locally:

```powershell id="o5br9r"
docker images opentripplanner/opentripplanner
```

If you see output similar to:

```text id="jsmu3a"
REPOSITORY                        TAG       IMAGE ID
opentripplanner/opentripplanner   latest    baad654ade69
```

the image is available and you can proceed to create the container.

If the command returns nothing, download the OTP image from Docker Hub first:

```powershell id="1gwv91"
docker pull opentripplanner/opentripplanner:latest
```

After the download, check again:

```powershell id="0cq4jt"
docker images opentripplanner/opentripplanner
```

Confirm that the OTP image is present.

> For formal paper experiments, pin a specific OTP version instead of relying on `latest` so the experimental environment remains reproducible.

---

## 3. Create the OTP Docker Container

Navigate to:

```text id="l646bh"
G:\HUJI-FMCDRT\FMC15_TravelAnalysis_V6\otp_jerusalem
```

Run:

```powershell id="7vrgf2"
docker run -d `
  --name otp `
  --restart unless-stopped `
  -p 8080:8080 `
  -e JAVA_TOOL_OPTIONS="-Xmx6g" `
  -v "G:\HUJI-FMCDRT\FMC15_TravelAnalysis_V6\otp_jerusalem\data:/var/opentripplanner" `
  opentripplanner/opentripplanner:latest `
  --load `
  --serve
```

This command:

* Creates a Docker container named `otp`;
* Mounts the local data directory containing `graph.obj`;
* Loads the existing `graph.obj`;
* Starts the OTP service locally on port `8080`.

If you use a pinned image version, for example:

```text id="65q2hf"
opentripplanner/opentripplanner:2.10.0_xxx
```

replace:

```text id="225azh"
opentripplanner/opentripplanner:latest
```

in the command above with the corresponding pinned version tag.

---

## 4. Verify That OTP Has Started

List the running Docker containers:

```powershell id="bzlirc"
docker ps
```

Normally, you should see a container named:

```text id="5p8lhl"
otp
```

with a port mapping that includes:

```text id="ku8l1g"
0.0.0.0:8080->8080/tcp
```

View the OTP logs:

```powershell id="y8nenl"
docker logs -f otp
```

If the logs contain no obvious messages such as:

```text id="wl9h90"
ERROR
Exception
Failed
Unable to load graph
```

OTP has usually loaded `graph.obj` successfully.

Press:

```text id="3aej63"
Ctrl + C
```

to stop viewing the logs. This does not stop the OTP service.

---

## 5. OTP Service URLs

Local OTP service URL:

```text id="1ylfcm"
http://localhost:8080
```

GraphQL endpoint used by the DRT project:

```text id="v3frdm"
http://localhost:8080/otp/gtfs/v1
```

Example project configuration:

```yaml id="6a81t9"
otp:
  enabled: true
  graphql_endpoint: http://localhost:8080/otp/gtfs/v1
```

---

## 6. Routine Use

After creating the `otp` container for the first time, you do not need to run `docker run` again.

Start OTP:

```powershell id="8n4k0o"
docker start otp
```

Stop OTP:

```powershell id="ecwxxh"
docker stop otp
```

Restart OTP:

```powershell id="1hyyxk"
docker restart otp
```

Check its status:

```powershell id="nze6in"
docker ps
```

View the logs:

```powershell id="wxn5wb"
docker logs -f otp
```

---

As long as `graph.obj` has not changed, you do not need to rebuild the OTP graph or recreate the Docker container.
