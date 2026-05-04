from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def _sample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(7)
    indices = rng.choice(len(points), size=max_points, replace=False)
    indices.sort()
    return points[indices], colors[indices]


def write_point_cloud_viewer(
    pcd: o3d.geometry.PointCloud,
    output_path: str | Path,
    *,
    title: str = "Canopy 3D Viewer",
    max_points: int = 90000,
) -> Path:
    if pcd is None or pcd.is_empty():
        raise ValueError("Cannot write an interactive viewer for an empty point cloud.")

    output_path = Path(output_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
    else:
        colors = np.full(points.shape, 0.35, dtype=np.float32)

    points, colors = _sample_points(points, colors, max_points=max(1, int(max_points)))
    center = points.mean(axis=0)
    points = points - center
    span = float(np.linalg.norm(np.ptp(points, axis=0)))
    if span <= 1e-9:
        span = 1.0
    points = points / span

    payload = {
        "title": title,
        "points": np.round(points, 5).tolist(),
        "colors": np.clip(np.round(colors, 4), 0.0, 1.0).tolist(),
        "count": int(len(points)),
    }

    html = _viewer_html(title, payload)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _viewer_html(title: str, payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #101314;
      color: #eef2ed;
      font-family: Arial, Helvetica, sans-serif;
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
      cursor: grab;
      touch-action: none;
    }}
    canvas:active {{
      cursor: grabbing;
    }}
    .hud {{
      position: fixed;
      left: 16px;
      top: 14px;
      padding: 8px 10px;
      background: rgba(16, 19, 20, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.35;
      pointer-events: none;
    }}
  </style>
</head>
<body>
<canvas id="viewer"></canvas>
<div class="hud">
  <strong>{title}</strong><br>
  Drag to rotate. Wheel to zoom. Double click to reset.<br>
  Points: <span id="count"></span>
</div>
<script id="pcd-data" type="application/json">{data_json}</script>
<script>
const payload = JSON.parse(document.getElementById("pcd-data").textContent);
document.getElementById("count").textContent = payload.count.toLocaleString();

const canvas = document.getElementById("viewer");
const gl = canvas.getContext("webgl", {{ antialias: true }});
if (!gl) {{
  document.body.innerHTML = "<p style='padding:20px'>WebGL is not available in this browser.</p>";
}}

const vertexSource = `
attribute vec3 position;
attribute vec3 color;
uniform mat4 matrix;
uniform float pointSize;
varying vec3 vColor;
void main() {{
  gl_Position = matrix * vec4(position, 1.0);
  gl_PointSize = pointSize;
  vColor = color;
}}`;

const fragmentSource = `
precision mediump float;
varying vec3 vColor;
void main() {{
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  if (dot(uv, uv) > 1.0) discard;
  gl_FragColor = vec4(vColor, 1.0);
}}`;

function compileShader(type, source) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
    throw new Error(gl.getShaderInfoLog(shader));
  }}
  return shader;
}}

const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
  throw new Error(gl.getProgramInfoLog(program));
}}
gl.useProgram(program);

function flatten(items) {{
  const out = new Float32Array(items.length * 3);
  for (let i = 0; i < items.length; i++) {{
    out[i * 3] = items[i][0];
    out[i * 3 + 1] = items[i][1];
    out[i * 3 + 2] = items[i][2];
  }}
  return out;
}}

function bindAttribute(name, data) {{
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const location = gl.getAttribLocation(program, name);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
}}

bindAttribute("position", flatten(payload.points));
bindAttribute("color", flatten(payload.colors));

const matrixLocation = gl.getUniformLocation(program, "matrix");
const pointSizeLocation = gl.getUniformLocation(program, "pointSize");

let rotationX = -0.85;
let rotationY = 0.0;
let zoom = 2.35;
let dragging = false;
let lastX = 0;
let lastY = 0;

function identity() {{
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
}}

function multiply(a, b) {{
  const out = new Array(16).fill(0);
  for (let row = 0; row < 4; row++) {{
    for (let col = 0; col < 4; col++) {{
      for (let k = 0; k < 4; k++) {{
        out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
      }}
    }}
  }}
  return out;
}}

function perspective(fovy, aspect, near, far) {{
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect,0,0,0,
    0,f,0,0,
    0,0,(far + near) * nf,-1,
    0,0,(2 * far * near) * nf,0
  ];
}}

function rotateX(angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];
}}

function rotateY(angle) {{
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1];
}}

function translate(z) {{
  const out = identity();
  out[14] = z;
  return out;
}}

function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
  gl.viewport(0, 0, canvas.width, canvas.height);
}}

function draw() {{
  resize();
  gl.clearColor(0.062, 0.074, 0.078, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const aspect = canvas.width / Math.max(canvas.height, 1);
  let matrix = perspective(Math.PI / 4, aspect, 0.05, 20);
  matrix = multiply(matrix, translate(-zoom));
  matrix = multiply(matrix, rotateX(rotationX));
  matrix = multiply(matrix, rotateY(rotationY));
  gl.uniformMatrix4fv(matrixLocation, false, new Float32Array(matrix));
  gl.uniform1f(pointSizeLocation, Math.max(2.0, Math.min(5.0, 5.2 - zoom)));
  gl.drawArrays(gl.POINTS, 0, payload.count);
  requestAnimationFrame(draw);
}}

canvas.addEventListener("pointerdown", (event) => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", (event) => {{
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  rotationY += dx * 0.008;
  rotationX += dy * 0.008;
}});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  zoom = Math.max(0.75, Math.min(7.0, zoom + event.deltaY * 0.0015));
}}, {{ passive: false }});
canvas.addEventListener("dblclick", () => {{
  rotationX = -0.85;
  rotationY = 0.0;
  zoom = 2.35;
}});
window.addEventListener("resize", resize);
draw();
</script>
</body>
</html>
"""
