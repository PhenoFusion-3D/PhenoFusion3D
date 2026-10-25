# Container validation evidence

Prepared 26 October 2026. This record describes the reproducible Linux environment
check, separately from the user's confirmation that the native software works on
the lab machine.

- Build revision: `21f3930e18d4f1ee7e8e4cfb011432561241a64f`.
- Successful run: [GitHub Actions 40246710829](https://github.com/Adithya-Rama/PhenoFusion3D/actions/runs/40246710829).
- Platform: Linux amd64 on the Ubuntu CI runner.
- Image ID: `sha256:d11c41b7e9e405764e518b4226e0968c603b29c3501073553a74a3fd4a03849b`.
- Default image check runs with `--network none`, without camera or gantry devices.
- It starts Qt offscreen, imports the analysis stack, checks a synthetic 3D
  transformation and OpenCV conversion, then emits a JSON pass result.
- Research workspace and hyperspectral command help entry points also run offline.
- The workflow rejects missing/invalid JSON or nonzero exit status before export.

The final upload folder supplies `container/CONTAINER_VERIFICATION.json`, the
downloaded smoke result, resolved dependency versions, image inspection and the
image export checksum. Use that receipt to establish local export integrity.
Subsequent documentation-only commits do not change this image's build revision.

## Superseded check

Run 40245717222 is not valid acceptance evidence: Qt failed to initialise, while
the original shell pipeline hid that failure and left an empty smoke result.
The replacement adds the missing Qt runtime libraries, fails on pipeline errors,
and parses/asserts the result. The rejected image is not in the upload package.

## Remaining physical acceptance

This does not demonstrate full application interaction, physical camera capture,
gantry movement or a real-plant reconstruction inside the container on the lab
machine. Record those tests independently if required for D6 acceptance. Continue
using the confirmed native lab installation until container display/hardware
configuration has been tested. See [container use](../../install/container/README.md)
and the [release record](RELEASE_RECORD.md).
