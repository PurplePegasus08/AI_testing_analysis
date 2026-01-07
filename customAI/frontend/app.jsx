import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import axios from "axios";
// ----------  YOUR ORIGINAL ICONS  ----------
import { LayoutDashboard, Database, BarChart2, BrainCircuit, Settings, Sparkles, Moon, Sun, LogOut, Upload, Send, User, Bot } from "lucide-react";

const WS_URL   = "ws://localhost:8000/api/ws";
const UPLOAD_URL = "http://localhost:8000/api/upload";

// ----------  YOUR ORIGINAL COLOURS  ----------
const dark = {
  bg: "#111827",  card:"#1f2937",  border:"#374151",
  text:"#f3f4f6",  sub:"#9ca3af",  accent:"#3b82f6"
};
const light = {
  bg:"#ffffff",   card:"#f9fafb",  border:"#e5e7eb",
  text:"#111827",  sub:"#6b7280",  accent:"#2563eb"
};

function App(){
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentView, setCurrentView] = useState("DASHBOARD");
  const [sessionId, setSessionId] = useState("");
  const [chat, setChat] = useState([{who:"bot", text:"Upload a CSV to start"}]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);

  const theme = darkMode ? dark : light;

  // ----------  SIDEBAR  ----------
  const Nav = ({ico, label, id})=>(
    <button onClick={()=>setCurrentView(id)} className={`flex items-center gap-3 px-3 py-2.5 rounded-xl w-full transition-all ${currentView===id?"bg-blue-500/10 text-blue-400":"text-gray-400 hover:bg-white/5"}`}>
      {ico} <span className={`${sidebarOpen?"":"hidden"} text-sm font-semibold`}>{label}</span>
    </button>
  );

  // ----------  FILE UPLOAD  ----------
  const onFile = async(e)=>{
    const file = e.target.files[0];
    if(!file) return;
    addChat("bot","Uploading…");
    const fd = new FormData(); fd.append("file", file);
    const {data} = await axios.post(UPLOAD_URL, fd);
    setSessionId(data.sessionId);
    addChat("bot", data.stats);
    setCurrentView("CHAT");
  };

  // ----------  WEBSOCKET CHAT  ----------
  const addChat = (who,text)=> setChat(c=>[...c,{who,text}]);
  useEffect(()=>{
    if(!sessionId) return;
    const ws = new WebSocket(`${WS_URL}/${sessionId}`);
    ws.onopen = ()=>{ setConnected(true); addChat("bot","Connected – ask me anything"); };
    ws.onmessage = (e)=> addChat("bot", JSON.parse(e.data).text);
    ws.onclose  = ()=> setConnected(false);
    window.sendChat = (txt)=> ws.send(JSON.stringify({type:"chat", text:txt}));
    return ()=> ws.close();
  },[sessionId]);

  const sendMsg = ()=>{
    if(!input.trim()) return;
    addChat("user", input);
    window.sendChat(input);
    setInput("");
  };

  // ----------  UI  ----------
  return (
    <div className="flex h-screen" style={{background:theme.bg, color:theme.text}}>
      {/* -------- Sidebar -------- */}
      <aside className={`flex flex-col ${sidebarOpen?"w-64":"w-20"} px-4 py-6 border-r transition-all`} style={{borderColor:theme.border, background:theme.card}}>
        <div className="flex items-center gap-3 mb-8">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 grid place-items-center"><Sparkles className="text-white w-5 h-5"/></div>
          {sidebarOpen&&<h1 className="font-bold text-lg tracking-tight">InsightFlow</h1>}
        </div>
        <nav className="flex-1 space-y-2">
          <Nav ico={<LayoutDashboard size={20}/>} id="DASHBOARD" label="Dashboard"/>
          <Nav ico={<Database size={20}/>}           id="DATA"      label="Data Studio"/>
          <Nav ico={<BarChart2 size={20}/>}          id="VIS"       label="Visualization"/>
          <Nav ico={<BrainCircuit size={20}/>}       id="CHAT"      label="AI Insights"/>
        </nav>
        <div className="mt-auto space-y-2">
          <button onClick={()=>setDarkMode(!darkMode)} className="flex items-center gap-3 px-3 py-2.5 rounded-xl w-full text-gray-400 hover:bg-white/5">
            {darkMode?<Moon size={20}/>:<Sun size={20}/>} {sidebarOpen&&<span className="text-sm font-semibold">Theme</span>}
          </button>
          <button className="flex items-center gap-3 px-3 py-2.5 rounded-xl w-full text-red-400 hover:bg-red-500/10">
            <LogOut size={20}/> {sidebarOpen&&<span className="text-sm font-semibold">Logout</span>}
          </button>
        </div>
      </aside>

      {/* -------- Main -------- */}
      <main className="flex-1 flex flex-col p-6">
        {/* ----- Dashboard ----- */}
        {currentView==="DASHBOARD"&&<>
          <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
          <div className="grid grid-cols-3 gap-4">
            <StatCard label="Rows" value="—"/>
            <StatCard label="Cols" value="—"/>
            <StatCard label="Missing" value="—"/>
          </div>
        </>}

        {/* ----- Data Studio (only preview) ----- */}
        {currentView==="DATA"&&<>
          <h2 className="text-2xl font-bold mb-4">Data Studio</h2>
          <div className="rounded-2xl border p-4" style={{borderColor:theme.border, background:theme.card}}>
            <div className="text-sm text-gray-400">First 5 rows arrive here after upload – expand as you wish</div>
          </div>
        </>}

        {/* ----- Visualization (placeholder) ----- */}
        {currentView==="VIS"&&<>
          <h2 className="text-2xl font-bold mb-4">Visualization</h2>
          <div className="rounded-2xl border p-4" style={{borderColor:theme.border, background:theme.card}}>
            <div className="text-sm text-gray-400">Chart configs from backend will land here</div>
          </div>
        </>}

        {/* ----- AI Chat (your original look) ----- */}
        {currentView==="CHAT"&&<ChatPanel/>}
      </main>
    </div>
  );

  // ----------  Re-usable components ----------
  function StatCard({label,value}){
    return (
      <div className="rounded-2xl border p-4" style={{borderColor:theme.border, background:theme.card}}>
        <div className="text-sm text-gray-400">{label}</div>
        <div className="text-2xl font-bold mt-1">{value}</div>
      </div>
    );
  }

  function ChatPanel(){
    return (
      <div className="flex flex-col h-full">
        <h2 className="text-2xl font-bold mb-4">AI Insights</h2>
        <div className="flex-1 rounded-2xl border p-4 overflow-y-auto" style={{borderColor:theme.border, background:theme.card}}>
          {chat.map((m,i)=>(
            <div key={i} className={`flex gap-3 mb-3 ${m.who==="user"?"flex-row-reverse":""}`}>
              <div className={`w-8 h-8 rounded-lg grid place-items-center text-white ${m.who==="user"?"bg-blue-600":"bg-gray-700"}`}>
                {m.who==="user"?<User size={16}/>:<Bot size={16}/>}
              </div>
              <pre className={`px-4 py-2 rounded-2xl text-sm ${m.who==="user"?"bg-blue-600 text-white":"bg-gray-800 text-gray-200"}`}>
                {m.text}
              </pre>
            </div>
          ))}
        </div>
        <div className="flex gap-2 mt-4">
          <input
            value={input}
            onChange={(e)=>setInput(e.target.value)}
            onKeyUp={(e)=>e.key==="Enter"&&sendMsg()}
            disabled={!connected}
            placeholder="ask anything…"
            className="flex-1 px-4 py-2 rounded-xl border focus:outline-none focus:ring-2 focus:ring-blue-500"
            style={{borderColor:theme.border, background:theme.card, color:theme.text}}
          />
          <button
            onClick={sendMsg}
            disabled={!connected}
            className="px-4 py-2 rounded-xl bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 flex items-center gap-2"
          >
            <Send size={16}/> Send
          </button>
        </div>
      </div>
    );
  }
}

// ----------  mount  ----------
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
