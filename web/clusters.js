(() => {
  const el = id => document.getElementById(id);
  const canvas = el('clusterCanvas'), ctx = canvas.getContext('2d');
  const toggles = [...document.querySelectorAll('.attribute-switches input')];
  let data, active, bits = 7, selected = 0, pinned = false;
  let width = 1, height = 1, zoom = 1, panX = 0, panY = 0, points = [], drag = null;
  let categoryColors = {}, neighbors = new Set();
  const label = () => ['Palette', 'Pattern', 'Shape'].filter((_, j) => bits & (1 << j)).join(' + ');
  function reset() { zoom = 1; panX = panY = 0; draw(); }
  function coordinates(i) { return [points[i][0]*zoom+panX, points[i][1]*zoom+panY]; }
  function resize() {
    width = canvas.clientWidth; height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width*ratio; canvas.height = height*ratio;
    ctx.setTransform(ratio,0,0,ratio,0,0);
    if (!active) return;
    const xs = active.xy.map(p=>p[0]), ys = active.xy.map(p=>p[1]);
    const loX = Math.min(...xs), loY = Math.min(...ys);
    const rangeX = Math.max(...xs)-loX || 1, rangeY = Math.max(...ys)-loY || 1;
    const scale = Math.min((width-50)/rangeX,(height-50)/rangeY);
    points = active.xy.map(p=>[(p[0]-loX-rangeX/2)*scale+width/2,(p[1]-loY-rangeY/2)*scale+height/2]);
    reset();
  }
  function draw() {
    ctx.clearRect(0,0,width,height);
    if (!data || !active) return;
    data.items.forEach((item,i)=>{
      const [x,y]=coordinates(i);
      ctx.globalAlpha = neighbors.has(i) || i===selected ? 1 : .6;
      ctx.fillStyle=categoryColors[item.category]; ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
    });
    ctx.globalAlpha=1;
    for (const i of [...neighbors, selected]) {
      const [x,y]=coordinates(i);ctx.strokeStyle=i===selected?'#172438':'#e19231';ctx.lineWidth=i===selected?3:2;
      ctx.beginPath();ctx.arc(x,y,i===selected?8:6,0,Math.PI*2);ctx.stroke();
    }
  }
  function inspect(index) {
    if (!data || !active) return;
    selected=index; const item=data.items[index];
    el('clusterReference').src=`/images/${item.id}.jpg`;
    el('clusterTitle').textContent=`${item.style} · ${item.color}`;
    el('clusterMeta').textContent=item.category;
    el('unpinPoint').hidden=!pinned;
    el('neighborExplanation').textContent=`${label()} distance · lower is closer. Equal weights for selected attributes; one result per product, excluding this product's other views.`;
    const root=el('clusterNeighbors');root.replaceChildren();
    neighbors=new Set(active.neighbors[index].map(pair=>pair[0]));
    active.neighbors[index].forEach(([j,distance])=>{
      const target=data.items[j], button=document.createElement('button');button.className='cluster-neighbor';
      const img=document.createElement('img');img.src=`/images/${target.id}.jpg`;img.alt=target.name;img.loading='lazy';
      const name=document.createElement('span');name.textContent=`${target.category} · ${target.style}`;
      const score=document.createElement('strong');score.textContent=`Distance ${distance.toFixed(4)}`;
      button.append(img,name,score);button.onclick=()=>{pinned=true;inspect(j);};root.append(button);
    });draw();
  }
  function switchMap() {
    bits=toggles.reduce((v,t)=>v+(t.checked?Number(t.value):0),0);
    toggles.forEach(t=>t.disabled=t.checked && toggles.filter(x=>x.checked).length===1);
    if (!data) return;
    active=data.maps[String(bits)];
    el('mapTitle').textContent=label();
    el('mapStatus').textContent=`${data.items.length.toLocaleString()} catalog views · ${label()} · Equal attribute weights · Same selected item across maps`;
    resize();inspect(selected);
  }
  toggles.forEach(t=>t.addEventListener('change',switchMap));
  function hit(x,y) {
    let result=-1,best=100;
    points.forEach((_,i)=>{const [px,py]=coordinates(i);const d=(x-px)**2+(y-py)**2;if(d<best){best=d;result=i;}});
    return result;
  }
  canvas.addEventListener('pointerdown',e=>{drag={x:e.offsetX,y:e.offsetY,px:panX,py:panY,moved:false};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{
    if(drag){const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;if(drag.moved){panX=drag.px+dx;panY=drag.py+dy;draw();}return;}
    if(pinned)return;const i=hit(e.offsetX,e.offsetY);if(i>=0&&i!==selected)inspect(i);
  });
  canvas.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const i=hit(e.offsetX,e.offsetY);if(i>=0){pinned=!(pinned&&i===selected);inspect(i);}}drag=null;});
  canvas.addEventListener('pointercancel',()=>{drag=null;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();const next=Math.max(.5,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));panX=e.offsetX-(e.offsetX-panX)*next/zoom;panY=e.offsetY-(e.offsetY-panY)*next/zoom;zoom=next;draw();},{passive:false});
  el('resetMap').onclick=reset;
  el('randomPoint').onclick=()=>{if(data){pinned=true;inspect(Math.floor(Math.random()*data.items.length));}};
  el('unpinPoint').onclick=()=>{pinned=false;inspect(selected);};
  new ResizeObserver(resize).observe(el('mapFrame'));
  fetch('clusters-data.json').then(r=>{if(!r.ok)throw Error('Map data unavailable');return r.json();}).then(value=>{
    data=value;
    const categories=[...new Set(data.items.map(x=>x.category))].sort();
    categories.forEach((category,i)=>{
      const color=`hsl(${(i*137.508)%360} 48% 47%)`;categoryColors[category]=color;
      const tag=document.createElement('span'), dot=document.createElement('i');dot.style.background=color;tag.append(dot,document.createTextNode(category));el('mapLegend').append(tag);
    });switchMap();
  }).catch(error=>{el('mapStatus').textContent=`Could not load maps: ${error.message}. Reload to retry.`;});
})();
