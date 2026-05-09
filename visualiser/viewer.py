from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Standalone HTML viewer writer (used by canopy reconstruction)
# ---------------------------------------------------------------------------

def _sample_points(
    points: np.ndarray, colors: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
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
    max_points: int = 90_000,
) -> Path:
    """Write a self-contained WebGL HTML file that lets the user orbit the point cloud."""
    if pcd is None or pcd.is_empty():
        raise ValueError("Cannot write a viewer for an empty point cloud.")

    output_path = Path(output_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = (
        np.asarray(pcd.colors, dtype=np.float32)
        if pcd.has_colors()
        else np.full(points.shape, 0.35, dtype=np.float32)
    )

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
    output_path.write_text(_viewer_html(title, payload), encoding="utf-8")
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
    html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#101314;color:#eef2ed;font-family:Arial,sans-serif}}
    canvas{{display:block;width:100vw;height:100vh;cursor:grab;touch-action:none}}
    canvas:active{{cursor:grabbing}}
    .hud{{position:fixed;left:16px;top:14px;padding:8px 10px;background:rgba(16,19,20,.72);border:1px solid rgba(255,255,255,.16);border-radius:6px;font-size:13px;line-height:1.35;pointer-events:none}}
  </style>
</head>
<body>
<canvas id="v"></canvas>
<div class="hud"><strong>{title}</strong><br>Drag=rotate &bull; Wheel=zoom &bull; Dbl-click=reset<br>Points: <span id="cnt"></span></div>
<script id="d" type="application/json">{data_json}</script>
<script>
const p=JSON.parse(document.getElementById("d").textContent);
document.getElementById("cnt").textContent=p.count.toLocaleString();
const cv=document.getElementById("v"),gl=cv.getContext("webgl",{{antialias:true}});
if(!gl){{document.body.innerHTML="<p style='padding:20px'>WebGL unavailable.</p>";}}
function sh(t,src){{const s=gl.createShader(t);gl.shaderSource(s,src);gl.compileShader(s);return s;}}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,`attribute vec3 pos;attribute vec3 col;uniform mat4 M;uniform float PS;varying vec3 vc;void main(){{gl_Position=M*vec4(pos,1.);gl_PointSize=PS;vc=col;}}`));
gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 vc;void main(){{vec2 u=gl_PointCoord*2.-1.;if(dot(u,u)>1.)discard;gl_FragColor=vec4(vc,1.);}}` ));
gl.linkProgram(prog);gl.useProgram(prog);
function flat(a){{const f=new Float32Array(a.length*3);for(let i=0;i<a.length;i++){{f[i*3]=a[i][0];f[i*3+1]=a[i][1];f[i*3+2]=a[i][2];}}return f;}}
function buf(name,data){{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);const l=gl.getAttribLocation(prog,name);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,0,0);}}
buf("pos",flat(p.points));buf("col",flat(p.colors));
const ML=gl.getUniformLocation(prog,"M"),PL=gl.getUniformLocation(prog,"PS");
let rX=-0.85,rY=0,zoom=2.35,drag=false,lx=0,ly=0;
function id(){{return[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];}}
function mul(a,b){{const o=new Array(16).fill(0);for(let r=0;r<4;r++)for(let c=0;c<4;c++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o;}}
function persp(f,asp,n,fa){{const t=1/Math.tan(f/2),nf=1/(n-fa);return[t/asp,0,0,0,0,t,0,0,0,0,(fa+n)*nf,-1,0,0,(2*fa*n)*nf,0];}}
function rx(a){{const c=Math.cos(a),s=Math.sin(a);return[1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1];}}
function ry(a){{const c=Math.cos(a),s=Math.sin(a);return[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1];}}
function tr(z){{const o=id();o[14]=z;return o;}}
function resize(){{const d=Math.min(window.devicePixelRatio||1,2),w=Math.floor(cv.clientWidth*d),h=Math.floor(cv.clientHeight*d);if(cv.width!==w||cv.height!==h){{cv.width=w;cv.height=h;}}gl.viewport(0,0,cv.width,cv.height);}}
function draw(){{resize();gl.clearColor(.062,.074,.078,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);const asp=cv.width/Math.max(cv.height,1);let M=persp(Math.PI/4,asp,.05,20);M=mul(M,tr(-zoom));M=mul(M,rx(rX));M=mul(M,ry(rY));gl.uniformMatrix4fv(ML,false,new Float32Array(M));gl.uniform1f(PL,Math.max(2,Math.min(5,5.2-zoom)));gl.drawArrays(gl.POINTS,0,p.count);requestAnimationFrame(draw);}}
cv.addEventListener("pointerdown",e=>{{drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId);}});
cv.addEventListener("pointermove",e=>{{if(!drag)return;rY+=(e.clientX-lx)*.008;rX+=(e.clientY-ly)*.008;lx=e.clientX;ly=e.clientY;}});
cv.addEventListener("pointerup",()=>drag=false);
cv.addEventListener("wheel",e=>{{e.preventDefault();zoom=Math.max(.75,Math.min(7,zoom+e.deltaY*.0015));}},{{passive:false}});
cv.addEventListener("dblclick",()=>{{rX=-0.85;rY=0;zoom=2.35;}});
window.addEventListener("resize",resize);
draw();
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Interactive Open3D viewer (used by the PyQt application)
# ---------------------------------------------------------------------------


class PointCloudViewer:
    """
    Non-blocking Open3D visualiser window.
    Updated from the Qt controller thread via update().
    """

    def __init__(self):
        self.vis      = None
        self._started = False
        self._has_geom = False

    def start(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='PhenoFusion3D - Point Cloud',
            width=900, height=700
        )
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.point_size = 1.5
        self._started   = True
        self._has_geom  = False

    def update(self, pcd):
        if not self._started or self.vis is None:
            return
        if pcd is None or pcd.is_empty():
            return
        if not self._has_geom:
            self.vis.add_geometry(pcd)
            self._has_geom = True
            self.vis.reset_view_point(True)
            vc = self.vis.get_view_control()
            vc.set_front([0.0, -0.3, -1.0])
            vc.set_up([0.0, -1.0, 0.0])
            vc.set_zoom(0.5)
        else:
            self.vis.update_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if self.vis:
            self.vis.destroy_window()
            self.vis      = None
            self._started = False