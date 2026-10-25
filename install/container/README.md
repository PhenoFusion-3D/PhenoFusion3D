# Recorded data container

This container packages the application and analysis dependencies for Linux
amd64. The existing lab installation remains the established native virtual
environment with host ROS and camera drivers. Do not replace it during delivery.
The container is a separate delivery item; only its build receipt establishes
whether an image has been produced and tested.

## Build and verify

From the repository root on a machine with a working Docker engine:

```sh
docker build -f install/container/Dockerfile --build-arg SOURCE_REVISION=$(git rev-parse HEAD) -t phenofusion3d:handover .
docker run --rm --network none phenofusion3d:handover
docker run --rm --network none phenofusion3d:handover python -m processing.hyperspectral --help
```

The default check imports the dependency stack, creates an offscreen Qt
application, and tests a synthetic point-cloud transformation. It does not access
cameras, move hardware, reproduce a plant result, or prove full lab acceptance.

Direct dependencies are pinned in `requirements.txt`. The complete resolved
environment is stored inside each image as `/opt/phenofusion3d/environment.txt`.
Keep the built image ID and export checksum; a rebuild may resolve different
transitive dependencies or base-image updates. The CI workflow records this
evidence and exports an image artifact. Download it before its retention expires
and add it to stakeholder-controlled storage; an expiring CI link is not custody.

## Process local recordings

Use separate mounts for immutable input and writable outputs. On Linux, choose
an existing output directory owned by the current user:

```sh
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/generated/container-output",dst=/results \
  phenofusion3d:handover python -m processing.research_workspace --help
```

Replace the help invocation with the documented workflow and container-visible
paths. Input configuration files containing absolute Windows paths must be
copied and adapted explicitly; never rewrite the originals. Validate expected
outputs against the reference workflow before operational use.

## Export and recover

```sh
docker save phenofusion3d:handover | gzip -1 > PhenoFusion3D_recorded_data_image.tar.gz
sha256sum PhenoFusion3D_recorded_data_image.tar.gz
gunzip -c PhenoFusion3D_recorded_data_image.tar.gz | docker load
```

`docker load -i PhenoFusion3D_recorded_data_image.tar.gz` is also supported by
Docker. After load, rerun the check with `--network none`. Real data and exports
are external mounts, so replacing an image must not remove them.

## Hardware and display boundary

ROS, gantry services, USB permissions, camera firmware and physical safety
procedures remain host/lab responsibilities. No devices are passed through by
default. Qt defaults to offscreen mode for repeatable checks. Interactive display,
OpenGL and embedded browser behaviour require a separately tested host display
configuration. Do not disable host access controls or expose X11 broadly to make
a demonstration work. Use the confirmed native lab UI until container display
and hardware paths have a witnessed test record.

## Preparation status on 26 October 2026

The local Docker Desktop engine could not start because its inference-manager
socket failed to initialise. Source build files and a CI verification/export
workflow are provided. Consult the package release receipt for subsequent CI
results; absence of a successful receipt means the image remains unverified.
