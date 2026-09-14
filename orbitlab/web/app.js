(() => {
  "use strict";
  const model = window.ORBITLAB_DATA;
  if (!model) return;
  const $ = id => document.getElementById(id);
  const canvas = $("orbit-canvas");
  const ctx = canvas.getContext("2d");
  const R = model.meta.radius;
  let selected = 0, index = 0, playing = false;
  let yaw = -0.55, pitch = 0.32, zoom = 1, drag = null, lastAdvance = 0;
  let projectedTracks = [];

  const norm = v => Math.hypot(v[0], v[1], v[2]);
  const sample = (object=model.objects[selected], i=index) => object.samples[Math.min(i, object.samples.length-1)];
  const num = (v, n=6) => Number(v).toFixed(n);
  const vector = (v, n) => `[${v.map(x => `${x >= 0 ? "+" : ""}${x.toFixed(n)}`).join(", ")}]`;
  const met = seconds => {
    const h=String(Math.floor(seconds/3600)).padStart(2,"0");
    const m=String(Math.floor(seconds%3600/60)).padStart(2,"0");
    const s=String(Math.floor(seconds%60)).padStart(2,"0");
    return `T+${h}:${m}:${s}`;
  };

  function resize() {
    const dpr=Math.min(devicePixelRatio||1,2), box=canvas.getBoundingClientRect();
    canvas.width=Math.round(box.width*dpr); canvas.height=Math.round(box.height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0); return box;
  }
  function rotate(v) {
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const x=cy*v[0]+sy*v[2], z=-sy*v[0]+cy*v[2];
    return [x,cp*v[1]-sp*z,sp*v[1]+cp*z];
  }

  function render() {
    const box=resize(),w=box.width,h=box.height,cx=w/2,cy=h/2;
    const outer=selected===3?29000:9000, scale=Math.min(w,h)*.43/outer*zoom;
    const project=v=>{const p=rotate(v);return [cx+p[0]*scale,cy-p[1]*scale,p[2]];};
    const earthRadius=R*scale;
    ctx.clearRect(0,0,w,h); ctx.fillStyle="#000"; ctx.fillRect(0,0,w,h);

    const axisLength=outer*.76;
    [[axisLength,0,0,"x"],[0,axisLength,0,"y"],[0,0,axisLength,"z"]].forEach(([x,y,z,label])=>{
      const a=project([0,0,0]),b=project([x,y,z]);
      ctx.strokeStyle="#252a2d";ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
      ctx.fillStyle="#555d61";ctx.font="9px monospace";ctx.fillText(label,b[0]+4,b[1]-4);
    });

    // Far-side orbit arcs are drawn first, then occulted by the Earth.
    model.objects.forEach((object,oi)=>{
      if(selected!==3&&oi===3)return;
      ctx.strokeStyle=object.color+"24";ctx.lineWidth=.7;ctx.beginPath();
      object.samples.forEach((s,i)=>{const p=project(s[1]);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);});ctx.stroke();
    });

    ctx.fillStyle="#030607";ctx.beginPath();ctx.arc(cx,cy,earthRadius,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle="#536168";ctx.lineWidth=1;ctx.stroke();

    // Latitude/longitude curves are sampled on the reference sphere, not screen-space ellipses.
    const curves=[];
    for(let lat=-60;lat<=60;lat+=30){const c=[];for(let lon=0;lon<=360;lon+=3){const a=lat*Math.PI/180,b=lon*Math.PI/180;c.push([R*Math.cos(a)*Math.cos(b),R*Math.cos(a)*Math.sin(b),R*Math.sin(a)]);}curves.push(c);}
    for(let lon=0;lon<180;lon+=30){const c=[];for(let lat=-90;lat<=90;lat+=3){const a=lat*Math.PI/180,b=lon*Math.PI/180;c.push([R*Math.cos(a)*Math.cos(b),R*Math.cos(a)*Math.sin(b),R*Math.sin(a)]);}curves.push(c);}
    ctx.strokeStyle="#1d2529";ctx.lineWidth=.65;
    curves.forEach(curve=>{ctx.beginPath();let pen=false;curve.forEach(v=>{const p=project(v);if(p[2]>=0){pen?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);pen=true;}else pen=false;});ctx.stroke();});

    projectedTracks=[];
    model.objects.forEach((object,oi)=>{
      if(selected!==3&&oi===3)return;
      const points=[];let pen=false;ctx.beginPath();
      object.samples.forEach(s=>{const p=project(s[1]);points.push(p);const occulted=p[2]<0&&Math.hypot(p[0]-cx,p[1]-cy)<earthRadius;if(occulted){pen=false;return;}pen?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);pen=true;});
      ctx.strokeStyle=object.color+(oi===selected?"ff":"8c");ctx.lineWidth=oi===selected?1.35:.75;ctx.stroke();
      projectedTracks.push({oi,points});
    });

    model.objects.forEach((object,oi)=>{
      if(selected!==3&&oi===3)return;
      const p=project(sample(object)[1]);
      if(p[2]<0&&Math.hypot(p[0]-cx,p[1]-cy)<earthRadius)return;
      ctx.fillStyle=object.color;ctx.beginPath();ctx.arc(p[0],p[1],oi===selected?3:2,0,Math.PI*2);ctx.fill();
      if(oi===selected){ctx.font="9px monospace";ctx.fillStyle="#aeb5b8";ctx.fillText(object.id,p[0]+7,p[1]-6);}
    });

    const event=model.conjunction;
    if(Math.abs(index-event.sample)<=2){
      const a=model.objects.find(o=>o.id===event.a),b=model.objects.find(o=>o.id===event.b);
      const pa=project(sample(a,event.sample)[1]),pb=project(sample(b,event.sample)[1]);
      ctx.strokeStyle="#ffffff88";ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.setLineDash([]);
    }
  }

  function updateData() {
    const s=sample(),r=s[1],v=s[2],e=model.conjunction;
    $("d-time").textContent=`${s[0].toFixed(1)} s`;
    $("d-r").textContent=`${vector(r,3)} km`; $("d-v").textContent=`${vector(v,6)} km s⁻¹`;
    $("d-rmag").textContent=`${num(norm(r),3)} km`; $("d-vmag").textContent=`${num(norm(v),6)} km s⁻¹`;
    $("d-alt").textContent=`${num(norm(r)-R,3)} km`; $("d-a").textContent=`${num(s[3],6)} km`;
    $("d-e").textContent=Number(s[4]).toExponential(9); $("d-i").textContent=`${num(s[5],6)} deg`;
    $("d-raan").textContent=`${num(s[6],6)} deg`; $("d-argp").textContent=`${num(s[7],6)} deg`;
    $("d-nu").textContent=`${num(s[8],6)} deg`; $("d-energy").textContent=`${Number(s[9]).toExponential(9)} km² s⁻²`;
    $("d-rho").textContent=`${Number(s[10]).toExponential(9)} kg m⁻³`;
    $("d-mu").textContent=`${model.meta.mu} km³ s⁻²`; $("d-radius").textContent=`${model.meta.radius} km`;
    $("d-j2").textContent=Number(model.meta.j2).toExponential(8); $("d-step").textContent=`${model.meta.integration_step_s} s`;
    $("d-integrator").textContent=model.meta.integrator; $("d-pair").textContent=`${e.a} / ${e.b}`;
    $("d-tca").textContent=`${e.t_s.toFixed(1)} s`; $("d-miss").textContent=`${num(e.miss_km,6)} km`;
    $("d-vrel").textContent=`${num(e.relative_speed_km_s,6)} km s⁻¹`;
    $("clock").textContent=met(s[0]); $("timeline").value=index;
  }
  function update(){updateData();render();}
  function togglePanel(show){$("data-panel").hidden=!show;$("data-toggle").setAttribute("aria-expanded",String(show));}

  $("object-select").innerHTML=model.objects.map((o,i)=>`<option value="${i}">${o.id} · ${o.name}</option>`).join("");
  $("object-select").addEventListener("change",e=>{selected=Number(e.target.value);zoom=selected===3?.72:1;update();});
  $("data-toggle").addEventListener("click",()=>togglePanel($("data-panel").hidden));
  $("data-close").addEventListener("click",()=>togglePanel(false));
  $("timeline").max=model.objects[0].samples.length-1;
  $("timeline").addEventListener("input",e=>{index=Number(e.target.value);update();});
  $("play").addEventListener("click",()=>{playing=!playing;$("play").textContent=playing?"Ⅱ":"▶";});
  canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch};canvas.setPointerCapture(e.pointerId);canvas.classList.add("dragging");});
  canvas.addEventListener("pointermove",e=>{if(!drag)return;yaw=drag.yaw+(e.clientX-drag.x)*.006;pitch=Math.max(-1.25,Math.min(1.25,drag.pitch+(e.clientY-drag.y)*.006));render();});
  canvas.addEventListener("pointerup",()=>{drag=null;canvas.classList.remove("dragging");});
  canvas.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.4,Math.min(3.2,zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
  canvas.addEventListener("click",e=>{
    if(drag)return;
    let best={distance:12,oi:null};
    projectedTracks.forEach(track=>track.points.forEach(p=>{const d=Math.hypot(e.clientX-p[0],e.clientY-p[1]);if(d<best.distance)best={distance:d,oi:track.oi};}));
    if(best.oi!==null){selected=best.oi;$("object-select").value=selected;update();}
  });
  window.addEventListener("resize",render);

  function animate(now){if(playing&&now-lastAdvance>120){index=(index+1)%model.objects[0].samples.length;lastAdvance=now;update();}requestAnimationFrame(animate);}
  update();requestAnimationFrame(animate);
})();
