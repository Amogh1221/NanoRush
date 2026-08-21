'use client';
import { useChatStore } from '@/lib/store';
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Send, StopCircle, SquarePen } from 'lucide-react';

export default function ChatShell() {
  const { messages, isStreaming, addMessage, updateLastMessage, setStreaming, clearMessages } = useChatStore();
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Background wake-up ping on load
  useEffect(() => {
    // This simple request hits the health check endpoint.
    // Even if it fails due to CORS (which HF 503 pages often do),
    // the request itself signals Hugging Face to wake the space!
    fetch('https://amogh1221-nanorush.hf.space/').catch(() => {});
  }, []);

  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return;
    
    const userMsg = input.trim();
    setInput('');
    
    // Construct full history to send to backend
    const payloadMessages = [...messages, { role: 'user', content: userMsg }];
    
    addMessage({ id: Date.now().toString(), role: 'user', content: userMsg });
    addMessage({ id: (Date.now() + 1).toString(), role: 'assistant', content: '' });
    setStreaming(true);

    try {
      let res: Response | null = null;
      let isSleeping = false;

      // Retry loop for waking up the HF space (up to 150 seconds)
      for (let i = 0; i < 15; i++) {
        try {
          res = await fetch('https://amogh1221-nanorush.hf.space/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: payloadMessages }),
          });

          if (res.ok) {
            break; // Success!
          }
          if (res.status !== 503 && res.status !== 504) {
            break; // Some other error, don't retry for sleeping
          }
        } catch (e) {
          // HF 503 pages often lack CORS headers, causing fetch to throw a TypeError.
          // We catch it here and treat it as sleeping.
        }

        // If we reach here, the space is likely sleeping.
        if (!isSleeping) {
          isSleeping = true;
          updateLastMessage("⏳ *Backend is sleeping. Waking up model (this takes ~1-2 minutes)...*");
        }
        // Wait 10 seconds before trying again
        await new Promise(r => setTimeout(r, 10000));
      }

      if (!res || !res.ok) {
        updateLastMessage("Error connecting to NanoRush. The backend might be offline.");
        setStreaming(false);
        return;
      }

      // If we were sleeping and now succeeded, clear the loading message
      if (isSleeping) {
        updateLastMessage(""); 
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let fullText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                fullText += data.chunk;
                // Clean hallucination
                const cleanText = fullText.split("User:")[0].split("\\nUser")[0];
                updateLastMessage(cleanText);
              } catch (e) {}
            }
          }
        }
      }
    } catch (e) {
      updateLastMessage("Error connecting to NanoRush.");
    } finally {
      setStreaming(false);
    }
  };

  const composer = (
    <div className="w-full max-w-3xl mx-auto">
      <div className="relative flex items-center bg-[var(--bg-secondary)] border border-[var(--border-subtle)] rounded-xl shadow-sm p-1">
        <textarea 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }}
          placeholder="Message NanoRush..."
          className="w-full bg-transparent border-none outline-none resize-none px-4 py-3 text-[var(--text-primary)] max-h-32"
          rows={1}
        />
        <button 
          onClick={handleSubmit}
          disabled={!input.trim() || isStreaming}
          className="p-2 rounded-lg bg-[var(--bg-tertiary)] hover:bg-[var(--border-subtle)] text-[var(--text-primary)] transition-colors disabled:opacity-50 mr-2"
        >
          {isStreaming ? <StopCircle size={20} /> : <Send size={20} />}
        </button>
      </div>
      <div className="text-center text-xs text-[var(--text-secondary)] mt-3">
        NanoRush can make mistakes. Verify important info.
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-[100dvh] bg-[var(--bg-primary)] w-full">
      
      {/* Top Bar */}
      <header className="h-14 border-b border-[var(--border-subtle)] flex items-center px-4 justify-between sticky top-0 bg-[var(--bg-primary)] z-10 shrink-0">
        <div className="flex items-center gap-2">
          <h1 className="font-semibold text-[var(--text-primary)]">NanoRush</h1>
          <span className="text-xs bg-[var(--bg-tertiary)] px-2 py-0.5 rounded text-[var(--text-secondary)]">283M</span>
        </div>
        <div className="flex items-center gap-2 sm:gap-4">
          <div className="text-[10px] sm:text-xs text-[var(--text-secondary)] flex items-center gap-1 sm:gap-2">
            <span className="hidden sm:inline">Context Window:</span>
            <span>~{Math.min(100, Math.round((JSON.stringify(messages).length / 4) / 4096 * 100))}%</span>
            <div className="w-12 sm:w-24 h-1.5 sm:h-2 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
               <div className="h-full bg-[var(--accent)]" style={{ width: `${Math.min(100, (JSON.stringify(messages).length / 4) / 4096 * 100)}%` }}></div>
            </div>
          </div>
          <button 
            onClick={clearMessages}
            disabled={isStreaming}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)] transition-colors disabled:opacity-50"
          >
            <SquarePen size={16} />
            <span className="hidden sm:inline">New Chat</span>
          </button>
        </div>
      </header>

      {/* Message List */}
      <main className={`flex-1 overflow-y-auto p-4 md:p-8 ${messages.length === 0 ? 'flex flex-col justify-center items-center' : ''}`}>
        {messages.length === 0 ? (
          <div className="w-full flex flex-col items-center justify-center">
            <h2 className="text-2xl sm:text-3xl font-semibold text-[var(--text-primary)] text-center px-4">What can I help you with?</h2>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-2xl p-4 ${msg.role === 'user' ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)] rounded-br-sm' : 'bg-transparent text-[var(--text-primary)]'}`}>
                  {msg.role === 'assistant' ? (
                    <div className="prose prose-invert max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                        {msg.content || '▍'}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </main>

      {/* Composer (Bottom Docked) */}
      <div className="p-4 bg-gradient-to-t from-[var(--bg-primary)] to-transparent shrink-0">
        {composer}
      </div>
    </div>
  );
}
