"""Create a self-contained WebGL point-cloud review; no online libraries."""
import argparse,base64,json
from pathlib import Path
import numpy as np,open3d as o3d

def encode_cloud(path,limit=1200000):
 p=o3d.io.read_point_cloud(str(path))
 if not len(p.points):raise ValueError(f'Empty cloud: {path}')
 xyz=np.asarray(p.points).copy()*[1,-1,-1];rgb=np.uint8(np.clip(np.asarray(p.colors)*255,0,255))
 center=(np.percentile(xyz,.1,axis=0)+np.percentile(xyz,99.9,axis=0))/2;xyz-=center
 stride=max(1,int(np.ceil(len(xyz)/limit)));xyz=xyz[::stride].astype('<f4');rgb=rgb[::stride]
 xyz/=max(float(np.max(np.ptp(xyz,axis=0))),1e-8)
 return dict(count=len(xyz),original_count=len(p.points),xyz=base64.b64encode(xyz.tobytes()).decode(),rgb=base64.b64encode(rgb.tobytes()).decode(),span=float(np.max(np.ptp(xyz,axis=0))))

def run(args):
 out=Path(args.output);summary=json.loads(Path(args.summary).read_text())
 data=[encode_cloud(Path(args.cloud))]
 photo='data:image/png;base64,'+base64.b64encode(Path(args.photo).read_bytes()).decode()
 html=TEMPLATE.replace('__CLOUDS__',json.dumps(data)).replace('__SUMMARY__',json.dumps(summary)).replace('__PHOTO__',photo)
 out.write_text(html,encoding='utf-8');print(f'{out}: {out.stat().st_size:,} bytes',flush=True)

TEMPLATE=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Plant · reconstruction review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#061621;color:#eef6f7;font:15px system-ui,sans-serif}header{padding:24px 30px 16px;border-bottom:1px solid #28404d;display:flex;justify-content:space-between;gap:20px;align-items:center}h1{font-size:25px;margin:3px 0 6px;font-weight:580}.eyebrow{color:#8fb9ba;font-size:11px;text-transform:uppercase;letter-spacing:2px}.sub{color:#9cbbc8;font-size:13px}main{display:grid;grid-template-columns:minmax(0,1fr) 285px;height:calc(100vh - 113px);min-height:620px}.stage{position:relative;min-width:0;background:radial-gradient(ellipse at 45% 25%,#12425b,#04141d 80%)}canvas{width:100%;height:100%;display:block;touch-action:none}.hint{position:absolute;left:24px;bottom:18px;color:#90aeba;font-size:12px;pointer-events:none}.badge{position:absolute;left:24px;top:20px;color:#a8cbc6;font-size:12px;pointer-events:none}aside{padding:22px;background:#0a1d29;border-left:1px solid #29404b;overflow-y:auto}.label{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#96b3be;margin:22px 0 10px}.label:first-child{margin-top:0}button,select,a.download{border:1px solid #355464;background:#102b3c;color:#e7f3f5;border-radius:6px;padding:9px 11px;font:inherit;cursor:pointer}button:hover,a.download:hover{background:#1c4051}select{width:100%;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.grid button{font-size:12px}input[type=range]{width:100%;accent-color:#69ceaf}.row{display:flex;justify-content:space-between;color:#b4cbd5;font-size:12px;margin:10px 0 3px}p{font-size:12px;line-height:1.6;color:#a8c1cd}.stat{font-size:24px;color:#e0f2ed;font-variant-numeric:tabular-nums}.source{display:block;width:100%;border-radius:5px;border:1px solid #365262;cursor:zoom-in}.download{display:block;text-align:center;text-decoration:none;font-size:12px!important;margin:8px 0}.check{display:flex;align-items:center;gap:8px;font-size:12px;color:#b6cbd3;margin-top:15px}dialog{padding:10px;background:#092333;border:1px solid #507381;max-width:95vw}dialog img{max-width:90vw;max-height:80vh}dialog::backdrop{background:#000b}#error{padding:30px;color:#ffd0c6;display:none}footer{font-size:11px;color:#8da9b5;margin-top:20px;line-height:1.5}@media(max-width:750px){header{padding:16px}h1{font-size:21px}main{display:block;height:auto;min-height:0}.stage{height:62vh;min-height:400px}aside{border-left:0;border-top:1px solid #355464}.source{max-width:350px}.grid{grid-template-columns:repeat(3,1fr)}}
</style>
<header><div><div class="eyebrow">PhenoFusion3D · offline reconstruction</div><h1>Plant reconstruction review</h1><div class="sub">Offline reconstruction · inspect all sides before measurements</div></div><button id="reset">Reset view</button></header>
<main><section class="stage"><canvas id="view" aria-label="Interactive 3D plant point cloud"></canvas><div id="error"></div><div class="badge" id="badge"></div><div class="hint">Drag to rotate · scroll to zoom · Shift + drag to pan</div></section><aside>
<div class="label">Result</div><select id="model"><option value="0">Reconstructed point cloud</option></select>
<div class="label">Viewpoint</div><div class="grid"><button data-y="180" data-e="25">Front</button><button data-y="0" data-e="25">Back</button><button data-y="90" data-e="25">Left</button><button data-y="270" data-e="25">Right</button><button data-y="0" data-e="90">Top</button><button data-y="180" data-e="-35">Below</button></div>
<label class="check"><input id="spin" type="checkbox">Rotate automatically</label>
<div class="label">Display</div><label class="row" for="size"><span>Point size</span><span id="sizeval">2.0</span></label><input id="size" type="range" min="1" max="5" value="2" step=".1">
<label class="row" for="light"><span>Brightness</span><span id="lightval">1.35×</span></label><input id="light" type="range" min=".7" max="2.2" value="1.35" step=".05">
<div class="label">Reconstruction</div><div id="count" class="stat"></div><p id="details"></p>
<div class="label">Original reference photograph</div><img class="source" id="source" src="__PHOTO__" alt="Original reference photograph"><p>Click the photograph to inspect the source plant.</p>
<a class="download" href="plant_upright.ply" download>Download upright point cloud</a><a class="download" href="RECONSTRUCTION_REPORT.md">Read the evidence and limitations</a>
<footer>See the evidence report for the reconstruction method and its limits. Hidden undersides and remaining gaps are visible when rotated. Display brightness and point size do not alter the exported geometry.</footer>
</aside></main><dialog id="photo"><img src="__PHOTO__" alt="Full original frame"><br><button id="close">Close photograph</button></dialog>
<script>
const DATA=__CLOUDS__,SUMMARY=__SUMMARY__;
const canvas=document.getElementById('view'),gl=canvas.getContext('webgl',{antialias:true,alpha:true,preserveDrawingBuffer:true});
if(!gl){document.getElementById('error').style.display='block';document.getElementById('error').textContent='WebGL is unavailable in this browser. The downloadable PLY remains available.';throw Error('WebGL unavailable')}
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 p;attribute vec3 c;uniform vec3 right;uniform vec3 up;uniform vec3 toward;uniform vec2 scale;uniform vec2 pan;uniform float size;uniform float brightness;varying vec3 color;void main(){gl_Position=vec4(dot(p,right)*scale.x+pan.x,dot(p,up)*scale.y+pan.y,-dot(p,toward)*.7,1.);gl_PointSize=size;color=clamp(c*brightness,0.,1.);}`));
gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 color;void main(){if(length(gl_PointCoord-.5)>.5)discard;gl_FragColor=vec4(color,1.);}`));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const loc={};for(const n of ['right','up','toward','scale','pan','size','brightness'])loc[n]=gl.getUniformLocation(program,n);
function bytes(s){const t=atob(s),b=new Uint8Array(t.length);for(let i=0;i<t.length;i++)b[i]=t.charCodeAt(i);return b}
const buffers=DATA.map(d=>{const p=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,p);gl.bufferData(gl.ARRAY_BUFFER,bytes(d.xyz),gl.STATIC_DRAW);const c=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,c);gl.bufferData(gl.ARRAY_BUFFER,bytes(d.rgb),gl.STATIC_DRAW);return {p,c,count:d.count}});
const pa=gl.getAttribLocation(program,'p'),ca=gl.getAttribLocation(program,'c');gl.enableVertexAttribArray(pa);gl.enableVertexAttribArray(ca);gl.enable(gl.DEPTH_TEST);gl.clearColor(0,0,0,0);
let yaw=Math.PI,pitch=25*Math.PI/180,zoom=1,pan=[0,0],current=0,last=0;
function reset(){yaw=Math.PI;pitch=25*Math.PI/180;zoom=1;pan=[0,0]}
function info(){document.getElementById('count').textContent=DATA[current].original_count.toLocaleString()+' points';document.getElementById('badge').textContent=SUMMARY.minimum_support_views===null?'Sensor depth → lab ICP':'RGB stereo → coloured ICP → visibility-checked surface';document.getElementById('details').textContent=current===0?(SUMMARY.minimum_support_views===null?`${SUMMARY.frames.length} input views. Sensor-depth ICP; no per-point support-vote validation. Inspect scene background and edges before traits.`:`${SUMMARY.frames.length} fused views · at least ${SUMMARY.minimum_support_views} supporting views per point. Metric accuracy has not yet been validated against manual measurements.`):'Earlier reference-based sensor-depth fusion. Rotate it to compare the stretched edges and disconnected regions.'}
document.getElementById('model').onchange=e=>{current=Number(e.target.value);info()};document.getElementById('reset').onclick=reset;
for(const b of document.querySelectorAll('[data-y]'))b.onclick=()=>{yaw=Number(b.dataset.y)*Math.PI/180;pitch=Number(b.dataset.e)*Math.PI/180;pan=[0,0]};
for(const id of ['size','light'])document.getElementById(id).oninput=e=>document.getElementById(id+'val').textContent=Number(e.target.value).toFixed(id==='size'?1:2)+(id==='light'?'×':'');
let drag=null;canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);drag=[e.clientX,e.clientY]};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(e.shiftKey){pan[0]+=dx/canvas.clientWidth*2;pan[1]-=dy/canvas.clientHeight*2}else{yaw-=dx*.007;pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,pitch+dy*.007))}drag=[e.clientX,e.clientY]};canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(8,zoom*Math.exp(-e.deltaY*.001)))},{passive:false});
document.getElementById('source').onclick=()=>document.getElementById('photo').showModal();document.getElementById('close').onclick=()=>document.getElementById('photo').close();
function draw(time){const ratio=Math.min(devicePixelRatio||1,2),w=Math.round(canvas.clientWidth*ratio),h=Math.round(canvas.clientHeight*ratio);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h)}if(document.getElementById('spin').checked&&!drag)yaw+=(time-last)*.00015;last=time;
const a=yaw,b=pitch,r=[Math.cos(a),Math.sin(a),0],u=[-Math.sin(a)*Math.sin(b),Math.cos(a)*Math.sin(b),Math.cos(b)],t=[Math.sin(a)*Math.cos(b),-Math.cos(a)*Math.cos(b),Math.sin(b)];
gl.uniform3fv(loc.right,r);gl.uniform3fv(loc.up,u);gl.uniform3fv(loc.toward,t);const span=Math.max(DATA[0].span,.4)*1.22;gl.uniform2f(loc.scale,2*zoom/span*Math.min(h/w,1),2*zoom/span*Math.min(w/h,1));gl.uniform2fv(loc.pan,pan);gl.uniform1f(loc.size,Number(document.getElementById('size').value)*ratio);gl.uniform1f(loc.brightness,Number(document.getElementById('light').value));
const d=buffers[current];gl.bindBuffer(gl.ARRAY_BUFFER,d.p);gl.vertexAttribPointer(pa,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,d.c);gl.vertexAttribPointer(ca,3,gl.UNSIGNED_BYTE,true,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.drawArrays(gl.POINTS,0,d.count);requestAnimationFrame(draw)}info();requestAnimationFrame(draw);
</script></html>'''

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['cloud','summary','photo','output']:p.add_argument('--'+name,required=True)
 run(p.parse_args())
