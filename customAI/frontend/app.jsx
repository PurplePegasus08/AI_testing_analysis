import React, { useState, useEffect, useRef, useMemo } from "react";
import ReactDOM from "react-dom/client";
import axios from "axios";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import html2canvas from "html2canvas";

// ----------  COLORS  ----------
const dark = { bg:"#111827", card:"#1f2937", border:"#374151", text:"#f3f4f6", sub:"#9ca3af", accent:"#3b82f6" };
const light = { bg:"#ffffff", card:"#f9fafb", border:"#e5e7eb", text:"#111827", sub:"#6b7280", accent:"#2563eb" };

// ----------  TYPES  ----------
type DataRow = Record<string, unknown>;
type ChartConfig = {
  id:string; title:string; type:"bar"|"line"|"area"|"pie"|"doughnut";
  xAxisKey:string; yAxisKeys:string[]; theme:"default"|"neon"|"pastel"|"dark"|"professional";
  aggregation:"sum"|"avg"|"count"|"min"|"max"; sortByValue:"none"|"asc"|"desc";
  stacked?:boolean; smoothCurve?:boolean; showLegend?:boolean; showGrid?:boolean;
  legendPosition?:"top"|"bottom"|"left"|"right"; columnFilters?:Record<string, unknown[]>;
  x?:number; y?:number; width?:number; height?:number; zIndex?:number; locked?:boolean;
};

type ChartRow = Record<string, string | number>;

// ----------  MAIN APP  ----------
export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentView, setCurrentView] = useState("DASHBOARD");
  const [sessionId, setSessionId] = useState("");
  const [chat, setChat] = useState<{who:string; text:string}[]>([{who:"bot", text:"Upload a CSV to start"}]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState<DataRow[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [vizConfig, setVizConfig] = useState<ChartConfig>({
    id:"default",title:"New Chart",type:"bar",xAxisKey:"",yAxisKeys:[],theme:"default",aggregation:"sum",sortByValue:"none",showLegend:false,showGrid:true,legendPosition:"top",smoothCurve:true,stacked:false
  });
  const [dashboardItems, setDashboardItems] = useState<ChartConfig[]>([]);

  const theme = darkMode ? dark : light;

  // ----------  WEBSOCKET  ----------
  const addChat = (who:string, text:string)=> setChat(c=>[...c,{who,text}]);
  useEffect(()=>{
    if (!sessionId) return;
    const ws = new WebSocket(`ws://localhost:8000/api/ws/${sessionId}`);
    ws.onopen = ()=>{ setConnected(true); addChat("bot","Connected – ask me anything"); };
    ws.onmessage = (e)=>{
      const m = JSON.parse(e.data);
      if (m.type==="chat") addChat("bot",m.text);
      if (m.type==="viz")  setVizConfig(m.config);
      if (m.type==="dataUpdate") { setData(m.rows); setHeaders(m.cols); }
    };
    ws.onclose  = ()=> setConnected(false);
    window.sendChat = (txt:string)=> ws.send(JSON.stringify({type:"chat", text:txt}));
    return ()=> ws.close();
  },[sessionId]);

  // ----------  FILE UPLOAD  ----------
  const onFile = async (e:React.ChangeEvent<HTMLInputElement>)=>{
    const file = e.target.files?.[0];
    if (!file) return;
    addChat("bot","Uploading…");
    const fd = new FormData(); fd.append("file", file);
    const {data} = await axios.post("http://localhost:8000/api/upload", fd);
    setSessionId(data.sessionId);
    setData(data.preview);
    if (data.preview && data.preview.length > 0) {
      setHeaders(Object.keys(data.preview[0]));
    } else {
      setHeaders([]);
    }
    addChat("bot", data.stats);
    setCurrentView("DATA");
  };

  // ----------  CHAT  ----------
  const sendMsg = ()=>{
    if (!input.trim()) return;
    addChat("user", input);
    window.sendChat(input);
    setInput("");
  };

  // ----------  DASHBOARD  ----------
  const addToDashboard = (cfg:ChartConfig)=>{
    setDashboardItems(prev=>[...prev,{...cfg,id:Date.now().toString(),x:20,y:20,width:400,height:300,zIndex:10}]);
  };
  const removeFromDashboard = (id:string)=> setDashboardItems(prev=>prev.filter(i=>i.id!==id));
  const updateDashboardItem = (id:string, updates:Partial<ChartConfig>)=>
    setDashboardItems(prev=>prev.map(i=>i.id===id?{...i,...updates}:i));

  // ----------  DATA STUDIO  ----------
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [colFilters, setColFilters] = useState<Record<string,Set<unknown>>>({});
  const rowsPerPage = 50;

  const filtered = useMemo(()=>{
    let d = data;
    if (search) d = d.filter(r=>Object.values(r).some(v=>String(v).toLowerCase().includes(search.toLowerCase())));
    Object.entries(colFilters).forEach(([col,set])=>{ if(set.size) d = d.filter(r=>set.has(r[col])); });
    return d;
  },[data,search,colFilters]);

  const paginated = useMemo(()=>filtered.slice((page-1)*rowsPerPage, page*rowsPerPage),[filtered,page]);

  const toggleFilter = (col:string,val:unknown)=>{
    const s = new Set(colFilters[col]||[]);
    if (s.has(val)) s.delete(val); else s.add(val);
    setColFilters({...colFilters,[col]:s});
    setPage(1);
  };
  const clearFilter = (col:string)=>{
    const n={...colFilters}; delete n[col]; setColFilters(n); setPage(1);
  };

  const dataStats = useMemo(()=>{
    if (!data.length) return [];
    return headers.map(h=>{
      const vals = data.map(r=>r[h]).filter(v=>v!==null&&v!==""&&v!==undefined);
      const nums = vals.filter(v=>typeof v==="number"&&!Number.isNaN(v));
      const type = nums.length>vals.length*0.5?"number":"string";
      const missing=data.length-vals.length;
      const mean=nums.length?nums.reduce((a,b)=>a+b,0)/nums.length:0;
      const min=nums.length?Math.min(...nums):0;
      const max=nums.length?Math.max(...nums):0;
      return {h,type,missing,mean,min,max};
    });
  },[data,headers]);

  // ----------  VISUALIZATION  ----------
  const chartData = useMemo(()=>{
    if (!vizConfig.xAxisKey) return [];
    const agg = vizConfig.aggregation||"sum";
    const filters = vizConfig.columnFilters||{};
    let rows = data;
    Object.entries(filters).forEach(([col,vals])=>{ if(vals.length) rows = rows.filter(r=>vals.includes(r[col])); });
    const map=new Map<string,number[]>();
    rows.forEach(r=>{
      const k = String(r[vizConfig.xAxisKey]);
      if (!map.has(k)) map.set(k,[]);
      vizConfig.yAxisKeys.forEach(y=>{ const v=Number(r[y]); if(!isNaN(v)) map.get(k)!.push(v); });
    });
    const out:ChartRow[]=[];
    map.forEach((vals,name)=>{
      const row:ChartRow={name};
      vizConfig.yAxisKeys.forEach((y)=>{
        const nums=vals.filter(n=>!isNaN(n));
        if (agg==="sum") row[y]=nums.reduce((a,b)=>a+b,0);
        if (agg==="avg") row[y]=nums.length?nums.reduce((a,b)=>a+b,0)/nums.length:0;
        if (agg==="count") row[y]=nums.length;
        if (agg==="min") row[y]=nums.length?Math.min(...nums):0;
        if (agg==="max") row[y]=nums.length?Math.max(...nums):0;
      });
      out.push(row);
    });
    if (vizConfig.sortByValue!=="none") out.sort((a,b)=>{
      const v = Number(a[vizConfig.yAxisKeys[0]] ?? 0);
      const w = Number(b[vizConfig.yAxisKeys[0]] ?? 0);
      return vizConfig.sortByValue==="asc"?v-w:w-v;
    });
    return out;
  },[data,vizConfig]);

  const CHART_COLORS = {
    default:  ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"],
    neon:     ["#00ff00","#ff00ff","#00ffff","#ffff00","#ff0000"],
    pastel:   ["#a78bfa","#fbcfe8","#a5f3fc","#c7d2fe","#fde68a"],
    dark:     ["#4c1d95","#1e3a8a","#065f46","#92400e","#831843"],
    professional:["#1e293b","#64748b","#94a3b8","#cbd5e1","#f1f5f9"]
  };

  const exportChartPNG = async ()=>{
    const el = document.getElementById("chart-container");
    if (!el) return;
    const canvas = await html2canvas(el);
    const link = document.createElement("a");
    link.download = `${vizConfig.title || "chart"}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };
  const exportChartCSV = ()=>{
    if (!chartData.length) return;
    const headers = ["name", ...vizConfig.yAxisKeys];
    const csv = [headers.join(","), ...chartData.map(r=>headers.map(h=>r[h]).join(","))].join("\n");
    const blob = new Blob([csv],{type:"text/csv"});
    const link = document.createElement("a");
    link.download = `${vizConfig.title || "chart"}.csv`;
    link.href = URL.createObjectURL(blob);
    link.click();
  };

  // ----------  DASHBOARD CANVAS  ----------
  const canvasRef = useRef<HTMLDivElement>(null);
  const [dragId, setDragId] = useState<string|null>(null);
  const [resizeId, setResizeId] = useState<string|null>(null);
  const [grid, setGrid] = useState(true);
  const [locked, setLocked] = useState(false);
  const GRID_SIZE = 20;

  const handleAutoLayout = ()=>{
    const cols = 3, margin = 20, itemW = 380, itemH = 300;
    setDashboardItems(items=>items.map((item, idx)=>{
      const col = idx % cols, row = Math.floor(idx / cols);
      return {...item, x: margin + col * (itemW + margin), y: margin + row * (itemH + margin), width: itemW, height: itemH};
    }));
  };
  const exportBoardPNG = async ()=>{
    if (!canvasRef.current) return;
    const canvas = await html2canvas(canvasRef.current);
    const link = document.createElement("a");
    link.download = "board.png";
    link.href = canvas.toDataURL();
    link.click();
  };

  useEffect(()=>{
    const handleMouseMove = (e:MouseEvent)=>{
      if (!canvasRef.current || locked) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + canvasRef.current.scrollLeft;
      const y = e.clientY - rect.top  + canvasRef.current.scrollTop;
      if (dragId){
        const item = dashboardItems.find(i=>i.id===dragId);
        if (!item || item.locked) return;
        const newX = Math.round(x / GRID_SIZE) * GRID_SIZE;
        const newY = Math.round(y / GRID_SIZE) * GRID_SIZE;
        updateDashboardItem(dragId, {x:Math.max(0,newX), y:Math.max(0,newY)});
      }
      if (resizeId){
        const item = dashboardItems.find(i=>i.id===resizeId);
        if (!item || item.locked) return;
        const newW = Math.round((x - item.x) / GRID_SIZE) * GRID_SIZE;
        const newH = Math.round((y - item.y) / GRID_SIZE) * GRID_SIZE;
        updateDashboardItem(resizeId, {width:Math.max(100,newW), height:Math.max(100,newH)});
      }
    };
    const handleMouseUp = ()=>{ setDragId(null); setResizeId(null); };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return ()=>{ window.removeEventListener("mousemove", handleMouseMove); window.removeEventListener("mouseup", handleMouseUp); };
  },[dragId, resizeId, locked, dashboardItems]);

  // ----------  RENDER  ----------
  return (
    <div className="flex h-screen" style={{background:theme.bg, color:theme.text}}>
      {/* -------- Sidebar -------- */}
      <aside className={`flex flex-col ${sidebarOpen?"w-64":"w-20"} px-4 py-6 border-r transition-all`} style={{borderColor:theme.border, background:theme.card}}>
        <div className="flex items-center gap-3 mb-8">
          <div onClick={()=>setSidebarOpen(!sidebarOpen)} className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 grid place-items-center cursor-pointer hover:opacity-80 transition-opacity"><span className="text-white font-bold text-lg">✦</span></div>
          {sidebarOpen&&<h1 className="font-bold text-lg tracking-tight">InsightFlow</h1>}
        </div>
        <nav className="flex-1 space-y-2">
          {[
            {ico:"✦", id:"DASHBOARD", label:"Dashboard"},
            {ico:"📊", id:"DATA",      label:"Data Studio"},
            {ico:"📈", id:"VIS",       label:"Visualization"},
            {ico:"🧠", id:"CHAT",      label:"AI Insights"}
          ].map(n=>(
            <button key={n.id} onClick={()=>setCurrentView(n.id)} className={`flex items-center gap-3 px-3 py-2.5 rounded-xl w-full transition-all ${currentView===n.id?"bg-blue-500/10 text-blue-400":"text-gray-400 hover:bg-white/5"}`}>
              <span className="text-lg">{n.ico}</span>
              <span className={`${sidebarOpen?"":"hidden"} text-sm font-semibold`}>{n.label}</span>
            </button>
          ))}
        </nav>
        <div className="mt-auto space-y-2">
          <button onClick={()=>setDarkMode(!darkMode)} className="flex items-center gap-3 px-3 py-2.5 rounded-xl w-full text-gray-400 hover:bg-white/5">
            <span className="text-lg">{darkMode?"🌙":"☀️"}</span>
            {sidebarOpen&&<span className="text-sm font-semibold">Theme</span>}
          </button>
          <button className="flex items-center gap-3 px-3 py-2.5 rounded-xl w-full text-red-400 hover:bg-red-500/10">
            <span className="text-lg">↗</span>
            {sidebarOpen&&<span className="text-sm font-semibold">Logout</span>}
          </button>
        </div>
      </aside>

      {/* -------- Main -------- */}
      <main className="flex-1 flex flex-col p-6">
        {currentView==="DASHBOARD"&&<DashboardView/>}
        {currentView==="DATA"     &&<DataStudioView/>}
        {currentView==="VIS"      &&<VisualizationView/>}
        {currentView==="CHAT"     &&<ChatView theme={theme} chat={chat} input={input} setInput={setInput} sendMsg={sendMsg} connected={connected}/>}
      </main>
    </div>
  );

  // ----------  VIEWS  ----------
  function DashboardView(){
    return (
      <>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Dashboard</h2>
          <div className="flex items-center gap-2">
            <label className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5 cursor-pointer bg-blue-600 border-blue-600 text-white">
               Upload CSV
               <input type="file" accept=".csv" onChange={onFile} className="hidden" />
            </label>
            <button onClick={handleAutoLayout} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Auto Layout</button>
            <button onClick={()=>setGrid(!grid)} className={`px-3 py-1.5 rounded-lg border text-sm ${grid?"bg-blue-500/10 text-blue-400":"hover:bg-white/5"}`}>Grid</button>
            <button onClick={()=>setLocked(!locked)} className={`px-3 py-1.5 rounded-lg border text-sm ${locked?"bg-orange-500/10 text-orange-400":"hover:bg-white/5"}`}>{locked?"🔒":"🔓"}</button>
            <button onClick={exportBoardPNG} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Export PNG</button>
          </div>
        </div>
        <div ref={canvasRef} className={`flex-1 relative rounded-2xl border overflow-hidden ${grid?"canvas-grid":""}`} style={{borderColor:theme.border, background:theme.card}}>
          {dashboardItems.map(item=>(
            <div
              key={item.id}
              className="absolute flex flex-col group"
              style={{left:item.x,top:item.y,width:item.width,height:item.height,zIndex:item.zIndex||10,transition:dragId===item.id||resizeId===item.id?"none":"all .2s"}}
              onMouseDown={()=>{ if (locked||item.locked) return; setDragId(item.id); }}
              onContextMenu={(e)=>{ e.preventDefault(); if (item.locked) return; updateDashboardItem(item.id,{x:item.x+40,y:item.y+40}); }}
            >
              {/* header */}
              <div className="h-9 flex items-center justify-between px-3 bg-white/5 backdrop-blur rounded-t-2xl border-b border-white/10 cursor-move">
                <span className="text-xs font-bold uppercase tracking-widest truncate">{item.title}</span>
                <div className="flex items-center gap-1">
                  <button onClick={()=>updateDashboardItem(item.id,{locked:!item.locked})} className="px-2 py-0.5 rounded text-xs">{item.locked?"🔒":"🔓"}</button>
                  <button onClick={()=>removeFromDashboard(item.id)} className="px-2 py-0.5 rounded text-xs">✕</button>
                </div>
              </div>
              {/* chart */}
              <div className="flex-1 p-3">
                <DashboardChart item={item} data={data} isDarkMode={darkMode} />
              </div>
              {/* resize handle */}
              {!locked&&!item.locked&&(
                <div
                  onMouseDown={(e)=>{ e.stopPropagation(); setResizeId(item.id); }}
                  className="absolute bottom-1 right-1 w-4 h-4 cursor-se-resize"
                >
                  <div className="w-full h-full border-r-2 border-b-2 border-gray-400 rounded-sm"/>
                </div>
              )}
            </div>
          ))}
          {dashboardItems.length===0&&(
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <span className="text-4xl mb-2">📊</span>
              <p className="font-bold text-sm uppercase tracking-widest">Drop charts here</p>
            </div>
          )}
        </div>
      </>
    );
  }

  function DataStudioView(){
    return (
      <>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Data Studio</h2>
          <div className="flex items-center gap-2">
            <label className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5 cursor-pointer bg-blue-600 border-blue-600 text-white">
               Upload CSV
               <input type="file" accept=".csv" onChange={onFile} className="hidden" />
            </label>
            <input
              value={search}
              onChange={(e)=>setSearch(e.target.value)}
              placeholder="Search rows…"
              className="px-3 py-1.5 rounded-lg border text-sm"
              style={{borderColor:theme.border, background:theme.card}}
            />
            <button onClick={()=>setColFilters({})} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Clear Filters</button>
            <button onClick={()=>{ const csv=data.map(r=>headers.map(h=>r[h]).join(",")).join("\n"); const blob=new Blob([csv],{type:"text/csv"}); const link=document.createElement("a"); link.download="data.csv"; link.href=URL.createObjectURL(blob); link.click(); }} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Download CSV</button>
          </div>
        </div>
        <div className="flex-1 flex gap-4">
          {/* stats */}
          <div className="w-80 rounded-2xl border p-4 overflow-y-auto" style={{borderColor:theme.border, background:theme.card}}>
            <div className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-3">Column Stats</div>
            {dataStats.map(s=>(
              <div key={s.h} className="mb-3 p-2 rounded-lg bg-white/5">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold">{s.h}</span>
                  <span className="text-xs text-gray-400">{s.type} {s.missing>0&&<span className="text-red-400">({s.missing} missing)</span>}</span>
                </div>
                {s.type==="number"&&(
                  <div className="text-xs text-gray-400 mt-1">mean: {s.mean.toFixed(2)} • min: {s.min} • max: {s.max}</div>
                )}
                <div className="mt-2 flex flex-wrap gap-1">
                  {Array.from(new Set(data.map(r=>r[s.h]))).slice(0,20).map(v=>(
                    <button key={String(v)} onClick={()=>toggleFilter(s.h,v)} className={`px-2 py-0.5 rounded text-xs border ${(colFilters[s.h]?.has(v))?"bg-blue-500/10 text-blue-400":"hover:bg-white/5"}`} style={{borderColor:theme.border}}>{String(v)}</button>
                  ))}
                  <button onClick={()=>clearFilter(s.h)} className="px-2 py-0.5 rounded text-xs border hover:bg-white/5">Clear</button>
                </div>
              </div>
            ))}
          </div>
          {/* table */}
          <div className="flex-1 rounded-2xl border overflow-hidden" style={{borderColor:theme.border, background:theme.card}}>
            <div className="p-3 border-b" style={{borderColor:theme.border}}>
              <div className="text-sm text-gray-400">Showing {paginated.length} of {filtered.length} rows</div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-white/5">
                  <tr>
                    {headers.map(h=>(
                      <th key={h} className="px-3 py-2 text-left font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {paginated.map((r,i)=>(
                    <tr key={i} className="border-t" style={{borderColor:theme.border}}>
                      {headers.map(h=>(
                        <td key={h} className="px-3 py-2">{r[h]===null||r[h]===""||r[h]===undefined?<span className="text-gray-500">null</span>:String(r[h])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-3 border-t flex items-center justify-between text-xs" style={{borderColor:theme.border}}>
              <button disabled={page===1} onClick={()=>setPage(p=>p-1)} className="px-3 py-1 rounded border disabled:opacity-50">Prev</button>
              <span>Page {page} of {Math.ceil(filtered.length/rowsPerPage)||1}</span>
              <button disabled={page>=Math.ceil(filtered.length/rowsPerPage)} onClick={()=>setPage(p=>p+1)} className="px-3 py-1 rounded border disabled:opacity-50">Next</button>
            </div>
          </div>
        </div>
      </>
    );
  }

  function VisualizationView(){
    return (
      <>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Visualization</h2>
          <div className="flex items-center gap-2">
            <button onClick={()=>exportChartPNG()} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Export PNG</button>
            <button onClick={()=>exportChartCSV()} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Export CSV</button>
            <button onClick={()=>addToDashboard(vizConfig)} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-white/5">Add to Dashboard</button>
          </div>
        </div>
        <div className="flex flex-col lg:flex-row gap-4">
          {/* controls */}
          <div className="w-full lg:w-80 rounded-2xl border p-4 space-y-4" style={{borderColor:theme.border, background:theme.card}}>
            <div>
              <div className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">Chart Type</div>
              <div className="grid grid-cols-3 gap-2">
                {(["bar","line","area","pie","doughnut"] as ChartConfig["type"][]).map(t=>(
                  <button key={t} onClick={()=>setVizConfig({...vizConfig,type:t})} className={`px-3 py-2 rounded-xl border text-sm font-bold ${vizConfig.type===t?"bg-blue-600 text-white":"hover:bg-white/5"}`}>{t}</button>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">X Axis</div>
              <select value={vizConfig.xAxisKey} onChange={e=>setVizConfig({...vizConfig,xAxisKey:e.target.value})} className="w-full px-3 py-2 rounded-lg border" style={{borderColor:theme.border, background:theme.card}}>
                <option value="">Select column</option>
                {headers.map(h=><option key={h}>{h}</option>)}
              </select>
            </div>
            <div>
              <div className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">Y Axis (multi)</div>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {headers.map(h=>(
                  <label key={h} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={vizConfig.yAxisKeys.includes(h)} onChange={()=>setVizConfig({...vizConfig,yAxisKeys:vizConfig.yAxisKeys.includes(h)?vizConfig.yAxisKeys.filter(y=>y!==h):[...vizConfig.yAxisKeys,h]})} />
                    {h}
                  </label>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">Method</div>
              <select value={vizConfig.aggregation} onChange={e=>setVizConfig({...vizConfig,aggregation:e.target.value as ChartConfig["aggregation"]})} className="w-full px-3 py-2 rounded-lg border" style={{borderColor:theme.border, background:theme.card}}>
                <option value="sum">Sum</option><option value="avg">Average</option><option value="count">Count</option><option value="min">Min</option><option value="max">Max</option>
              </select>
            </div>
            <div>
              <div className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">Theme</div>
              <div className="grid grid-cols-3 gap-2">
                {(Object.keys(CHART_COLORS) as (keyof typeof CHART_COLORS)[]).map(t=>(
                  <button key={t} onClick={()=>setVizConfig({...vizConfig,theme:t})} className={`px-3 py-2 rounded-xl border text-sm font-bold ${vizConfig.theme===t?"bg-blue-600 text-white":"hover:bg-white/5"}`}>{t.slice(0,4)}</button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <label className="flex items-center gap-2"><input type="checkbox" checked={vizConfig.stacked||false} onChange={e=>setVizConfig({...vizConfig,stacked:e.target.checked})} /> Stacked</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={vizConfig.smoothCurve??true} onChange={e=>setVizConfig({...vizConfig,smoothCurve:e.target.checked})} /> Smooth</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={vizConfig.showLegend||false} onChange={e=>setVizConfig({...vizConfig,showLegend:e.target.checked})} /> Legend</label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={vizConfig.showGrid??true} onChange={e=>setVizConfig({...vizConfig,showGrid:e.target.checked})} /> Grid</label>
            </div>
          </div>
          {/* chart */}
          <div id="chart-container" className="flex-1 rounded-2xl border p-4" style={{borderColor:theme.border, background:theme.card}}>
            {vizConfig.xAxisKey?(
              <ResponsiveContainer width="100%" height="100%">
                {vizConfig.type==="bar"&&<BarChart data={chartData} margin={{top:10,right:10,left:10,bottom:10}}>
                  {vizConfig.showGrid&&<CartesianGrid strokeDasharray="3 3" stroke={darkMode?"rgba(255,255,255,.1)":"rgba(0,0,0,.1)"} />}
                  <XAxis dataKey="name" stroke={darkMode?"#9ca3af":"#6b7280"} tick={{fontSize:12}} />
                  <YAxis stroke={darkMode?"#9ca3af":"#6b7280"} tick={{fontSize:12}} />
                  <Tooltip contentStyle={{backgroundColor:darkMode?"#1f2937":"#ffffff",border:"none",borderRadius:8,fontSize:12,color:darkMode?"#fff":"#111"}} />
                  {vizConfig.showLegend&&<Legend wrapperStyle={{fontSize:12}} />}
                  {vizConfig.yAxisKeys.map((y,i)=>(
                    <Bar key={y} dataKey={y} stackId={vizConfig.stacked?"stack":undefined} fill={CHART_COLORS[vizConfig.theme][i%CHART_COLORS[vizConfig.theme].length]} radius={vizConfig.stacked?0:[4,4,0,0]} />
                  ))}
                </BarChart>}
                {vizConfig.type==="line"&&<LineChart data={chartData} margin={{top:10,right:10,left:10,bottom:10}}>
                  {vizConfig.showGrid&&<CartesianGrid strokeDasharray="3 3" stroke={darkMode?"rgba(255,255,255,.1)":"rgba(0,0,0,.1)"} />}
                  <XAxis dataKey="name" stroke={darkMode?"#9ca3af":"#6b7280"} tick={{fontSize:12}} />
                  <YAxis stroke={darkMode?"#9ca3af":"#6b7280"} tick={{fontSize:12}} />
                  <Tooltip contentStyle={{backgroundColor:darkMode?"#1f2937":"#ffffff",border:"none",borderRadius:8,fontSize:12,color:darkMode?"#fff":"#111"}} />
                  {vizConfig.showLegend&&<Legend wrapperStyle={{fontSize:12}} />}
                  {vizConfig.yAxisKeys.map((y,i)=>(
                    <Line key={y} type={vizConfig.smoothCurve?"monotone":"linear"} dataKey={y} stroke={CHART_COLORS[vizConfig.theme][i%CHART_COLORS[vizConfig.theme].length]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>}
                {vizConfig.type==="area"&&<AreaChart data={chartData} margin={{top:10,right:10,left:10,bottom:10}}>
                  {vizConfig.showGrid&&<CartesianGrid strokeDasharray="3 3" stroke={darkMode?"rgba(255,255,255,.1)":"rgba(0,0,0,.1)"} />}
                  <XAxis dataKey="name" stroke={darkMode?"#9ca3af":"#6b7280"} tick={{fontSize:12}} />
                  <YAxis stroke={darkMode?"#9ca3af":"#6b7280"} tick={{fontSize:12}} />
                  <Tooltip contentStyle={{backgroundColor:darkMode?"#1f2937":"#ffffff",border:"none",borderRadius:8,fontSize:12,color:darkMode?"#fff":"#111"}} />
                  {vizConfig.showLegend&&<Legend wrapperStyle={{fontSize:12}} />}
                  {vizConfig.yAxisKeys.map((y,i)=>(
                    <Area key={y} type={vizConfig.smoothCurve?"monotone":"linear"} stackId={vizConfig.stacked?"stack":undefined} dataKey={y} stroke={CHART_COLORS[vizConfig.theme][i%CHART_COLORS[vizConfig.theme].length]} fill={CHART_COLORS[vizConfig.theme][i%CHART_COLORS[vizConfig.theme].length]} fillOpacity={0.3} strokeWidth={2} />
                  ))}
                </AreaChart>}
                {(vizConfig.type==="pie"||vizConfig.type==="doughnut")&&<PieChart>
                  <Pie data={chartData} cx="50%" cy="50%" innerRadius={vizConfig.type==="doughnut"?"60%":"0%"} outerRadius="80%" paddingAngle={4} dataKey={vizConfig.yAxisKeys[0]} nameKey="name" stroke="none">
                    {chartData.map((_,i)=>(
                      <Cell key={`cell-${i}`} fill={CHART_COLORS[vizConfig.theme][i%CHART_COLORS[vizConfig.theme].length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{backgroundColor:darkMode?"#1f2937":"#ffffff",border:"none",borderRadius:8,fontSize:12,color:darkMode?"#fff":"#111"}} />
                  {vizConfig.showLegend&&<Legend wrapperStyle={{fontSize:12}} />}
                </PieChart>}
              </ResponsiveContainer>
            ):(
              <div className="h-full flex flex-col items-center justify-center text-gray-500">
                <span className="text-4xl mb-2">📊</span>
                <p className="font-bold text-sm uppercase tracking-widest">Map dimensions to render</p>
              </div>
            )}
          </div>
        </div>
      </>
    );
  }

  // ----------  DASHBOARD CHART COMPONENT  ----------
  function DashboardChart({item,data,isDarkMode}:{item:ChartConfig;data:DataRow[];isDarkMode:boolean}){
    const chartData = useMemo(()=>{
      if (!item.xAxisKey) return [];
      const agg = item.aggregation||"sum";
      const map=new Map<string,number[]>();
      data.forEach(r=>{
        const k=String(r[item.xAxisKey]);
        if (!map.has(k)) map.set(k,[]);
        item.yAxisKeys.forEach(y=>{ const v=Number(r[y]); if(!isNaN(v)) map.get(k)!.push(v); });
      });
      const out:ChartRow[]=[];
      map.forEach((vals,name)=>{
        const row:ChartRow={name};
        item.yAxisKeys.forEach((y)=>{
          const nums=vals.filter(n=>!isNaN(n));
          if (agg==="sum") row[y]=nums.reduce((a,b)=>a+b,0);
          if (agg==="avg") row[y]=nums.length?nums.reduce((a,b)=>a+b,0)/nums.length:0;
          if (agg==="count") row[y]=nums.length;
          if (agg==="min") row[y]=nums.length?Math.min(...nums):0;
          if (agg==="max") row[y]=nums.length?Math.max(...nums):0;
        });
        out.push(row);
      });
      if (item.sortByValue!=="none") out.sort((a,b)=>{
        const v = Number(a[item.yAxisKeys[0]] ?? 0);
        const w = Number(b[item.yAxisKeys[0]] ?? 0);
        return item.sortByValue==="asc"?v-w:w-v;
      });
      return out;
    },[data,item]);

    const colors = {
      default:  ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"],
      neon:     ["#00ff00","#ff00ff","#00ffff","#ffff00","#ff0000"],
      pastel:   ["#a78bfa","#fbcfe8","#a5f3fc","#c7d2fe","#fde68a"],
      dark:     ["#4c1d95","#1e3a8a","#065f46","#92400e","#831843"],
      professional:["#1e293b","#64748b","#94a3b8","#cbd5e1","#f1f5f9"]
    };

    return (
      <ResponsiveContainer width="100%" height="100%">
        {item.type==="bar"&&<BarChart data={chartData} margin={{top:10,right:10,left:10,bottom:10}}>
          <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode?"rgba(255,255,255,.1)":"rgba(0,0,0,.1)"} />
          <XAxis dataKey="name" stroke={isDarkMode?"#9ca3af":"#6b7280"} tick={{fontSize:10}} />
          <YAxis stroke={isDarkMode?"#9ca3af":"#6b7280"} tick={{fontSize:10}} />
          <Tooltip contentStyle={{backgroundColor:isDarkMode?"#1f2937":"#ffffff",border:"none",borderRadius:6,fontSize:10,color:isDarkMode?"#fff":"#111"}} />
          {item.yAxisKeys.map((y,i)=>(
            <Bar key={y} dataKey={y} stackId={item.stacked?"stack":undefined} fill={colors[item.theme][i%colors[item.theme].length]} radius={item.stacked?0:[2,2,0,0]} />
          ))}
        </BarChart>}
        {item.type==="line"&&<LineChart data={chartData} margin={{top:10,right:10,left:10,bottom:10}}>
          <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode?"rgba(255,255,255,.1)":"rgba(0,0,0,.1)"} />
          <XAxis dataKey="name" stroke={isDarkMode?"#9ca3af":"#6b7280"} tick={{fontSize:10}} />
          <YAxis stroke={isDarkMode?"#9ca3af":"#6b7280"} tick={{fontSize:10}} />
          <Tooltip contentStyle={{backgroundColor:isDarkMode?"#1f2937":"#ffffff",border:"none",borderRadius:6,fontSize:10,color:isDarkMode?"#fff":"#111"}} />
          {item.yAxisKeys.map((y,i)=>(
            <Line key={y} type={item.smoothCurve?"monotone":"linear"} dataKey={y} stroke={colors[item.theme][i%colors[item.theme].length]} strokeWidth={2} dot={false} />
          ))}
        </LineChart>}
        {item.type==="pie"&&<PieChart>
          <Pie data={chartData} cx="50%" cy="50%" outerRadius="80%" paddingAngle={4} dataKey={item.yAxisKeys[0]} nameKey="name" stroke="none">
            {chartData.map((_,i)=><Cell key={`cell-${i}`} fill={colors[item.theme][i%colors[item.theme].length]} />)}
          </Pie>
          <Tooltip contentStyle={{backgroundColor:isDarkMode?"#1f2937":"#ffffff",border:"none",borderRadius:6,fontSize:10,color:isDarkMode?"#fff":"#111"}} />
        </PieChart>}
        {(item.type==="area"||item.type==="doughnut")&&<div className="h-full flex items-center justify-center text-gray-500">More charts on Dashboard</div>}
      </ResponsiveContainer>
    );
  }
}

interface ChatViewProps {
  theme: { bg:string; card:string; border:string; text:string; sub:string; accent:string };
  chat: {who:string; text:string}[];
  input: string;
  setInput: (s:string)=>void;
  sendMsg: ()=>void;
  connected: boolean;
}

function ChatView({ theme, chat, input, setInput, sendMsg, connected }: ChatViewProps){
  return (
    <div className="flex flex-col h-full">
      <h2 className="text-2xl font-bold mb-4">AI Insights</h2>
      <div className="flex-1 rounded-2xl border p-4 overflow-y-auto" style={{borderColor:theme.border, background:theme.card}}>
        {chat.map((m,i)=>(
          <div key={i} className={`flex gap-3 mb-3 ${m.who==="user"?"flex-row-reverse":""}`}>
            <div className={`w-8 h-8 rounded-lg grid place-items-center text-white ${m.who==="user"?"bg-blue-600":"bg-gray-700"}`}>
              {m.who==="user"?"👤":"🤖"}
            </div>
            <pre className={`px-4 py-2 rounded-2xl text-sm ${m.who==="user"?"bg-blue-600 text-white":"bg-gray-800 text-gray-200"}`}>
              {m.text}
            </pre>
          </div>
        ))}
      </div>
      <div className="flex gap-3 mt-4">
        <input
          value={input}
          onChange={(e)=>setInput(e.target.value)}
          onKeyUp={(e)=>e.key==="Enter"&&sendMsg()}
          disabled={!connected}
          placeholder="ask anything…"
          className="flex-1 px-4 py-2 rounded-xl border focus:outline-none focus:ring-2 focus:ring-blue-500"
          style={{borderColor:theme.border, background:theme.card, color:theme.text}}
          autoFocus
        />
        <button
          onClick={sendMsg}
          disabled={!connected}
          className="px-4 py-2 rounded-xl bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
