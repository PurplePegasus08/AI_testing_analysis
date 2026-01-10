import React, { useState, useEffect, useCallback } from 'react';
import axios, { AxiosProgressEvent } from 'axios';
import { Sidebar } from './components/Sidebar';
import { DataStudio } from './views/DataStudio';
import { Visualization } from './views/Visualization';
import { AiInsights } from './views/AiInsights';
import { Dashboard } from './views/Dashboard';
import { AuthView } from './components/AuthView';
import { SettingsModal } from './components/SettingsModal';
import { AppView, DataRow, ChartConfig, ChatMessage, DashboardItem, User } from './types';
import { CheckCircle2, Info, Sparkles } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000';

function App() {
  // State from copyfrom/App.tsx
  const [currentView, setCurrentView] = useState<AppView>(AppView.DASHBOARD);
  const [data, setData] = useState<DataRow[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [dashboardItems, setDashboardItems] = useState<DashboardItem[]>([]);
  const [activeFilters, setActiveFilters] = useState<Record<string, any[]>>({});
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'info'} | null>(null);
  
  // Auth State
  const [user, setUser] = useState<User | null>(() => {
    const savedUser = localStorage.getItem('insightflow_user');
    return savedUser ? JSON.parse(savedUser) : null;
  });

  // Theme State
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true;
  });

  // WebSocket State
  const [sessionId, setSessionId] = useState<string>("");
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: '0', role: 'model', content: "Systems synchronized. Intelligence engine initialized. How shall we interrogate the data today?" }
  ]);

  // Visualization Config State
  const [vizConfig, setVizConfig] = useState<ChartConfig>({
    id: 'default',
    title: 'Intelligence Report',
    type: 'bar',
    xAxisKey: '',
    yAxisKeys: [],
    theme: 'default',
    aggregation: 'sum',
  });

  // Data meta
  const [totalRows, setTotalRows] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const PREVIEW_LIMIT = 20;

  // Effects
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // WebSocket Connection
  useEffect(() => {
    if (!sessionId) return;
    const wsUrl = `${WS_BASE_URL}/api/ws/${sessionId}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => { 
      setConnected(true); 
      setNotification({ message: "Connected to Neural Core", type: "success" });
    };
    
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "chat") {
        setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', content: m.text }]);
      }
      if (m.type === "stats") {
         setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', content: m.text }]);
      }
      if (m.type === "dataUpdate") { 
        setData(m.rows); 
        setHeaders(m.cols); 
        setNotification({ message: "Data Synced with Core", type: "success" });
      }
      if (m.type === "error") {
        setNotification({ message: m.text, type: 'info' });
      }
      // Handle tool outputs if the backend sends them specifically, 
      // otherwise they might come as text or special formatted text.
    };
    
    ws.onclose = () => {
      setConnected(false);
      setNotification({ message: "Connection Lost", type: "info" });
    };

    // Expose send function to window or via ref if needed, but we use a prop
    (window as any).sendChat = (txt: string) => {
      try {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: "chat", text: txt}));
        else setNotification({ message: "Chat connection not open", type: 'info' });
      } catch (e: any) {
        console.error('WS send failed', e);
        setNotification({ message: 'Failed to send chat', type: 'info' });
      }
    };
    
    return () => ws.close();
  }, [sessionId]);

  const sendMessage = async (text: string): Promise<string | undefined> => {
    // Prefer WebSocket when connected
    if (connected && (window as any).sendChat) {
      try { (window as any).sendChat(text); return undefined; } catch (e) { console.error('WS send failed', e); }
    }

    if (!sessionId) {
      setNotification({ message: 'No active session. Upload first.', type: 'info' });
      return undefined;
    }

    try {
      setNotification({ message: 'Sending via HTTP fallback...', type: 'info' });
      const { data } = await axios.post(`${API_BASE_URL}/api/chat/${sessionId}`, { text }, { timeout: 120000 });
      if (data) {
        if (data.type === 'chat') return data.text;
        if (data.type === 'error') setNotification({ message: data.text, type: 'info' });
      }
    } catch (err: any) {
      console.error('Chat failed', err);
      setNotification({ message: err?.response?.data?.detail || err.message || 'Chat failed', type: 'info' });
    }
    return undefined;
  };

  // Handlers
  const handleLogin = (newUser: User) => {
    setUser(newUser);
    localStorage.setItem('insightflow_user', JSON.stringify(newUser));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('insightflow_user');
    setCurrentView(AppView.DASHBOARD);
    setSessionId("");
    setData([]);
    setHeaders([]);
  };

  const handleFileUpload = async (file: File) => {
    // Check backend first
    try {
      const ping = await axios.get(`${API_BASE_URL}/api/health`, { timeout: 3000 });
      if (!ping || ping.status !== 200) {
        setNotification({ message: 'Backend unreachable. Please start the server.', type: 'info' });
        return;
      }
    } catch (e) {
      console.error('Health check failed', e);
      setNotification({ message: 'Backend unreachable. Please start the server.', type: 'info' });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setNotification({ message: 'Uploading to Secure Enclave...', type: 'info' });

    try {
      const fd = new FormData();
      fd.append('file', file);

      const { data: resData } = await axios.post(`${API_BASE_URL}/api/upload`, fd, {
        timeout: 120000,
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          const loaded = (progressEvent && (progressEvent.loaded as number)) || 0;
          const total = (progressEvent && (progressEvent.total as number)) || 0;
          if (total > 0) {
            const pct = Math.round((loaded / total) * 100);
            setUploadProgress(pct);
            setNotification({ message: `Uploading... ${pct}%`, type: 'info' });
          }
        }
      });

      console.log('Upload response', resData);
      setSessionId(resData.sessionId);
      if (resData.preview && Array.isArray(resData.preview)) {
        const initialPreview = resData.preview.slice(0, PREVIEW_LIMIT);
        setData(initialPreview);
        if (initialPreview.length > 0) setHeaders(Object.keys(initialPreview[0]));
        if (resData.preview.length > PREVIEW_LIMIT) {
          setNotification({ message: `Preview limited to ${PREVIEW_LIMIT} rows. Click "Load 200 rows" in Data Hub to fetch more.`, type: 'info' });
        }
      }
      setTotalRows(resData.totalRows ?? null);
      setNotification({ message: 'Upload Complete', type: 'success' });
      setCurrentView(AppView.DATA);

      // Add initial stats message
      if (resData.stats) {
        setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', content: resData.stats }]);
      }
    } catch (err: any) {
      console.error('Upload failed', err);
      // Detailed error handling
      let msg = 'Upload Failed';
      if (err.code === 'ERR_NETWORK') {
        msg = 'Server unreachable. Is the backend running on port 8000?';
      } else if (err.response) {
        // Server responded with a status code outside 2xx
        msg = err.response.data?.detail || `Server Error: ${err.response.status}`;
      } else if (err.message) {
        msg = err.message;
      }
      setNotification({ message: msg, type: 'info' });
    } finally {
      setIsUploading(false);
      setUploadProgress(null);
    }
  };

  const fetchPreview = async (limit = 200) => {
    if (!sessionId) {
      setNotification({ message: 'No active session. Upload first.', type: 'info' });
      return;
    }
    try {
      setNotification({ message: `Fetching ${limit} rows...`, type: 'info' });
      const { data } = await axios.get(`http://localhost:8000/api/preview/${sessionId}?limit=${limit}`);
      if (data && data.rows) {
        setData(data.rows);
        if (data.rows.length > 0) setHeaders(Object.keys(data.rows[0]));
        setNotification({ message: `Loaded ${data.rows.length} rows`, type: 'success' });
      } else {
        setNotification({ message: 'No preview available', type: 'info' });
      }
    } catch (e: any) {
      console.error('Preview fetch failed', e?.response?.data || e.message || e);
      setNotification({ message: e?.response?.data?.detail || 'Preview failed', type: 'info' });
    }
  };

  const handleUpdateDashboardItem = (id: string, updates: Partial<DashboardItem>) => {
    setDashboardItems(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
  };

  const handleRemoveDashboardItem = (id: string) => {
    setDashboardItems(prev => prev.filter(item => item.id !== id));
  };

  const handleAddToDashboard = (config: ChartConfig) => {
    const newItem: DashboardItem = {
      ...config,
      id: Date.now().toString(),
      x: (dashboardItems.length % 3) * 400, // Simple layout logic
      y: Math.floor(dashboardItems.length / 3) * 300,
      width: 400,
      height: 300,
      zIndex: 10
    };
    setDashboardItems(prev => [...prev, newItem]);
    setNotification({ message: "Widget Added to Dashboard", type: "success" });
  };

  if (!user) {
    return <AuthView onLogin={handleLogin} isDarkMode={isDarkMode} onToggleTheme={() => setIsDarkMode(!isDarkMode)} />;
  }

  return (
    <div className={`flex h-screen bg-brand-50 dark:bg-surface-950 text-surface-900 dark:text-surface-100 font-sans overflow-hidden transition-colors duration-300`}>
      <Sidebar 
        currentView={currentView}
        user={user}
        isOpen={isSidebarOpen}
        isDarkMode={isDarkMode}
        onToggleTheme={() => setIsDarkMode(!isDarkMode)}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        onNavigate={setCurrentView}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onLogout={handleLogout}
      />
      <main className={`flex-1 flex flex-col h-full overflow-hidden transition-all duration-300 relative`}>
        {/* Notification Toast */}
        {notification && (
          <div className="absolute top-6 left-1/2 -translate-x-1/2 z-[100] animate-slide-up">
            <div className={`px-4 py-2 rounded-xl shadow-2xl backdrop-blur-xl border border-white/10 flex items-center gap-3 ${
              notification.type === 'success' 
                ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20' 
                : 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20'
            }`}>
              {notification.type === 'success' ? <CheckCircle2 className="w-4 h-4" /> : <Info className="w-4 h-4" />}
              <span className="text-xs font-bold tracking-wide uppercase">{notification.message}</span>
            </div>
          </div>
        )}

        {currentView === AppView.DASHBOARD && (
          <Dashboard 
            data={data}
            headers={headers}
            isDarkMode={isDarkMode}
            items={dashboardItems}
            onUpdateItem={handleUpdateDashboardItem}
            onRemoveItem={handleRemoveDashboardItem}
            onNavigateToData={() => setCurrentView(AppView.DATA)}
          />
        )}

        {currentView === AppView.DATA && (
          <DataStudio 
            data={data}
            headers={headers}
            activeFilters={activeFilters}
            setActiveFilters={setActiveFilters}
            onFileUpload={handleFileUpload}
            onCleanData={(col, op) => console.log("Clean", col, op)} // Implement cleaning via WebSocket later if needed
            onRequestPreview={(limit: number) => fetchPreview(limit)}
            totalRows={totalRows}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
          />
        )}

        {currentView === AppView.VISUALIZE && (
          <Visualization 
            data={data}
            headers={headers}
            config={vizConfig}
            setConfig={setVizConfig}
            onAddToDashboard={handleAddToDashboard}
            isDarkMode={isDarkMode}
            activeFilters={activeFilters}
            setActiveFilters={setActiveFilters}
          />
        )}

        {currentView === AppView.INSIGHTS && (
          <AiInsights 
            data={data}
            headers={headers}
            messages={messages}
            setMessages={setMessages}
            sendMessage={sendMessage}
            connected={connected}
            sessionId={sessionId}
            onUpdateVisualization={setVizConfig}
            onCleanData={(col, op) => console.log("Clean", col, op)}
            onAddToDashboard={handleAddToDashboard}
            onOpenStudio={() => setCurrentView(AppView.VISUALIZE)}
          />
        )}
      </main>

      <SettingsModal 
        isOpen={isSettingsOpen}
        isDarkMode={isDarkMode}
        onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
        onClose={() => setIsSettingsOpen(false)}
        user={user}
        onUpdateUser={(u) => { setUser(u); localStorage.setItem('insightflow_user', JSON.stringify(u)); }}
      />
    </div>
  );
}

export default App;

